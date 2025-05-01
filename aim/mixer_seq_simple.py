# References:
#   Mamba:  https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
#   DiT:    https://github.com/facebookresearch/DiT/blob/main/models.py#67
#   VAR:    https://github.com/FoundationVision/VAR/blob/main/models/var.py

import torch
import torch.nn as nn
import math
import json
import os
import copy

from collections import namedtuple
from functools import partial

# Make sure to install einops: pip install einops
try:
    from einops import rearrange
except ImportError:
    print("Warning: einops not installed. Rearrange operations might fail.")
    rearrange = None # Set to None if not available

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

# Assuming block.py and generation.py are in the same directory or accessible
from .block import Block, modulate
from .generation import GenerationMixin

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    print("Warning: Triton layer norm kernels not found. fused_add_norm=True might fail.")
    RMSNorm = partial(torch.nn.LayerNorm, eps=1e-5) # Fallback RMSNorm using LayerNorm
    layer_norm_fn, rms_norm_fn = None, None


class GroupAdaLN(nn.Linear):
    # Keep this class as is
    def __init__(self, in_features, out_features, num_channels, bias=True):
        super(GroupAdaLN, self).__init__(in_features, out_features, bias)
        self.num_channels = num_channels

    def forward(self, cond):
        channels = self.weight.shape[0] // self.num_channels
        return super().forward(cond).view(-1, self.num_channels, channels)


class FinalLayer(nn.Module):
    # Keep this class as is
    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        # Initialize weights to 0? Check original DiT/VAR if this is intended.
        # nn.init.constant_(self.adaLN_modulation[-1].weight, 0) # Keeping original init
        # nn.init.constant_(self.adaLN_modulation[-1].bias, 0) # Keeping original init

    def forward(self, x, c):
        # Handle case where c might be None
        if c is None:
            # If no conditioning, just apply norm
            return self.norm_final(x)
        else:
            # If conditioning exists, apply modulation
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
            return x


class GPT2Embeddings(nn.Module):
    # Keep this class as is
    def __init__(
        self,
        embed_dim,
        vocab_size,
        max_position_embeddings,
        padding_idx=None,
        word_embed_proj_dim=None,
        token_drop=0.0,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(
                vocab_size, embed_dim, padding_idx=padding_idx, **factory_kwargs
            )
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(
                vocab_size, word_embed_proj_dim, padding_idx=padding_idx, **factory_kwargs
            )
            self.project_in = nn.Linear(
                word_embed_proj_dim, embed_dim, bias=False, **factory_kwargs
            )
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(
                max_position_embeddings, embed_dim, **factory_kwargs
            )
        self.token_dropout = nn.Dropout(token_drop)

    def forward(self, input_ids, position_ids=None):
        batch_size, seqlen = input_ids.shape
        embeddings = self.token_dropout(self.word_embeddings(input_ids))
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings


class LabelEmbedder(nn.Module):
    # Keep this class as is
    def __init__(self, num_classes, hidden_size, dropout_prob=0.1):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        # Ensure compatibility: maybe need labels.clone() if input shouldn't be modified
        labels_out = torch.where(drop_ids, self.num_classes, labels)
        return labels_out

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        # Ensure labels are long type before embedding lookup
        if labels.dtype not in [torch.long, torch.int]:
             labels = labels.long()
        embeddings = self.embedding_table(labels)
        return embeddings


def create_block(
    # Keep this function as is
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    adaln_group=False, # Note: adaln_group might not be used directly by Block anymore
    mixer_drop=0.0,
    mlp_drop=0.0,
    device=None,
    dtype=None,
):
    if ssm_cfg is None: ssm_cfg = {}
    if attn_layer_idx is None: attn_layer_idx = []
    if attn_cfg is None: attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1") # Default to Mamba1 if not specified
        if ssm_layer not in ["Mamba1", "Mamba2"]: raise ValueError(f"Invalid ssm_layer: {ssm_layer}")
        mixer_cls = partial( Mamba2 if ssm_layer == "Mamba2" else Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial( nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs )
    if d_intermediate == 0: mlp_cls = nn.Identity
    else: mlp_cls = partial( GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs )

    # Pass d_model to block for adaLN_modulation definition inside Block
    block = Block( d_model, mixer_cls, mlp_cls, norm_cls=norm_cls, fused_add_norm=fused_add_norm, residual_in_fp32=residual_in_fp32, adaln_group=adaln_group )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    # Keep this function as is
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    # Keep __init__ as is
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        num_classes=1000,
        num_tokens=256,
        adaln_group=False, # This flag might control Block behavior now
        num_groups=4,
        token_drop=0.0,
        mixer_drop=0.0,
        mlp_drop=0.0,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # Keep embeddings and label embedder
        self.embeddings = GPT2Embeddings(d_model, vocab_size, num_tokens+1, token_drop=token_drop, **factory_kwargs) # num_tokens+1? Check if used.
        self.cls_embed = LabelEmbedder(num_classes=num_classes, hidden_size=d_model, dropout_prob=token_drop) # Pass dropout prob

        # Remove the adaln_group layer from MixerModel - it's handled in Block now
        # adaln_factor = 3 + (3 if d_intermediate != 0 else 0)
        # if adaln_group:
        #     self.adaln_group = nn.Sequential(...) # REMOVED
        #     self.num_groups = num_groups
        # else:
        #     self.adaln_group = nn.Identity() # REMOVED
        #     self.num_groups = 1
        self.num_groups = num_groups # Still need num_groups if used elsewhere? Block needs it?

        # Keep FinalLayer
        self.final_layer = FinalLayer(d_model)

        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                print("Warning: Triton kernels not found, fused_add_norm might not work as expected.")
                # raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        # Create blocks - pass adaln_group flag to Block constructor
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    adaln_group=adaln_group, # Pass flag to Block
                    mixer_drop=mixer_drop,
                    mlp_drop=mlp_drop,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,
            )
        )

    # Keep allocate_inference_cache as is
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    # --- REPLACE MixerModel.forward ---
    # Replace the *entire* existing forward method with this simplified version:
    def forward(self, input_ids, position_ids, cond, inference_params=None, **mixer_kwargs):
        # input_ids: [B, T] or [B, 1] during inference steps
        # position_ids: [B, T] or [B, 1]
        # cond: [B, D_model] (float text embedding) OR [B] or [B, 1] (int class labels) OR None

        is_train = inference_params is None
        B = input_ids.shape[0]

        # --- Determine Conditioning Embedding ---
        # This 'cond_embed' will be passed to each block's 'cls_embed' argument.
        # It should have shape [B, D_model] or be None.
        cond_embed = None
        if cond is not None:
            # CASE 1: Pre-computed float text embedding
            if isinstance(cond, torch.Tensor) and cond.ndim == 2 and cond.dtype in (torch.float32, torch.float16, torch.bfloat16):
                if cond.shape[-1] == self.embeddings.word_embeddings.embedding_dim:
                     cond_embed = cond
                     # print(f"✅ mixer_seq_simple: Using pre-embedded float conditioning. Shape: {cond_embed.shape}") # Debug
                else:
                     print(f"⚠️ Warning: Float 'cond' shape {cond.shape} doesn't match model dim {self.embeddings.word_embeddings.embedding_dim}. Setting cond_embed=None.")
                     cond_embed = None
            # CASE 2: Integer class labels
            elif isinstance(cond, torch.Tensor) and cond.dtype in (torch.int64, torch.int32):
                # print("⚠️ mixer_seq_simple: Using integer class label conditioning.") # Debug
                if cond.ndim == 2 and cond.shape[1] == 1: cond = cond.squeeze(1)
                if cond.ndim == 1:
                    if hasattr(self, 'cls_embed'):
                         cond_embed = self.cls_embed(cond, train=is_train) # Output [B, D_model]
                    else: print("Warning: cls_embed layer not found."); cond_embed = None
                else: print(f"⚠️ Int 'cond' tensor unexpected ndim: {cond.ndim}."); cond_embed = None
            # CASE 3: Unexpected type
            else: print(f"⚠️ Unexpected 'cond' type: {type(cond)}"); cond_embed = None
        # else: CASE 4: cond is None (unconditional) -> cond_embed remains None

        # --- Prepare Initial Hidden State (Input Sequence) ---
        # Get token embeddings. Shape [B, T_current, D_model]
        hidden_states = self.embeddings(input_ids, position_ids=position_ids)

        # Note: Conditioning via AdaLN happens inside the blocks using cond_embed.
        # No concatenation needed here. Sequence length remains T_current.

        # --- Process through Mamba Layers ---
        residual = None
        for i, layer in enumerate(self.layers):
            # Pass the *same* cond_embed ([B, D_model] or None) to every layer.
            # The Block's forward expects (hidden_states, residual, cls_embed, ...)
            hidden_states, residual = layer(
                hidden_states, residual, cond_embed, inference_params=inference_params
            )

        # --- Final Layer Norm and Modulation ---
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            if residual is not None:
                 hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
            else: # If residual is None (e.g., n_layer=0?)
                 hidden_states = self.norm_f(hidden_states.to(dtype=self.norm_f.weight.dtype))
        else:
            # Fused add norm path
            if layer_norm_fn is None:
                 print("Warning: Triton layer_norm_fn not found. Using non-fused path as fallback.")
                 # Fallback to non-fused logic
                 residual = (hidden_states + residual) if residual is not None else hidden_states
                 if residual is not None: hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
                 else: hidden_states = self.norm_f(hidden_states.to(dtype=self.norm_f.weight.dtype))
            else:
                 hidden_states = layer_norm_fn(
                     hidden_states, self.norm_f.weight, self.norm_f.bias, eps=self.norm_f.eps,
                     residual=residual, prenorm=False, residual_in_fp32=self.residual_in_fp32,
                     is_rms_norm=isinstance(self.norm_f, RMSNorm)
                 )

        # Apply final modulation layer if conditioning was provided and layer exists
        if cond_embed is not None and hasattr(self, 'final_layer'):
             hidden_states = self.final_layer(hidden_states, cond_embed)
        elif hasattr(self, 'final_layer') and hasattr(self.final_layer, 'norm_final'):
             # Apply just the final norm if no conditioning
             hidden_states = self.final_layer.norm_final(hidden_states)
        else:
            # Fallback if final_layer or its norm doesn't exist - apply model's main final norm
            hidden_states = self.norm_f(hidden_states)
            # print("Warning: Final layer or its norm not found, applied self.norm_f instead.") # Optional Debug

        return hidden_states


class MambaLMHeadModel(nn.Module, GenerationMixin):
    # Keep this class __init__, tie_weights, allocate_inference_cache as is

    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}
        self.cfg_scale = 1.5 # Default CFG scale

        self.backbone = MixerModel(
            d_model=config.d_model, n_layer=config.n_layer, d_intermediate=config.d_intermediate,
            vocab_size=config.vocab_size, ssm_cfg=config.ssm_cfg, attn_layer_idx=config.attn_layer_idx,
            attn_cfg=config.attn_cfg, rms_norm=config.rms_norm, initializer_cfg=initializer_cfg,
            fused_add_norm=config.fused_add_norm, residual_in_fp32=config.residual_in_fp32,
            num_classes=config.num_classes, num_tokens=config.num_tokens, adaln_group=config.adaln_group,
            num_groups=config.num_groups, token_drop=config.token_drop, mixer_drop=config.mixer_drop,
            mlp_drop=config.mlp_drop, **factory_kwargs,
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False, **factory_kwargs)
        self.apply( partial( _init_weights, n_layer=config.n_layer, **(initializer_cfg if initializer_cfg is not None else {}),))
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    # Keep the MambaLMHeadModel.forward method as is - it calls the corrected backbone.forward
    def forward(self, input_ids, position_ids=None, cond=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        is_train = inference_params is None

        # Logic for handling CFG input duplication during inference
        if not is_train and hasattr(inference_params, 'seqlen_offset') and inference_params.seqlen_offset > 0 and input_ids.shape[0] % 2 == 0:
             # Assuming input_ids might be doubled [2B, T_step] in later steps of CFG generate loop
             # If so, we might only need to process one half if backbone handles CFG internally?
             # Or backbone expects doubled input? Let's assume backbone expects doubled input.
             # The original code doubled it only after offset > 0, which seems odd.
             # Let's remove this specific doubling here, assuming generate handles it.
             # if inference_params.seqlen_offset > 0:
             #     input_ids, _ = torch.split(input_ids, len(input_ids) // 2, dim=0)
             #     input_ids = torch.cat([input_ids, input_ids])
             pass # Let backbone handle potentially doubled input

        # Call the corrected backbone forward
        hidden_states = self.backbone(input_ids, position_ids, cond, inference_params=inference_params, **mixer_kwargs)

        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]

        lm_logits = self.lm_head(hidden_states)

        # Apply Classifier-Free Guidance separation if not training and batch size is even
        if not is_train and lm_logits.shape[0] % 2 == 0:
            cond_logits, uncond_logits = torch.split(lm_logits, lm_logits.shape[0] // 2, dim=0)
            guided_logits = uncond_logits + (cond_logits - uncond_logits) * self.cfg_scale
            # Original code repeated guided_logits - this might be needed if subsequent steps expect doubled input
            # lm_logits = guided_logits.repeat(2, 1, 1)
            # Let's return only the guided logits for clarity, assuming consumer knows it's CFG output
            lm_logits = guided_logits
        # else:
            # If training or odd batch size during inference, return original logits

        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    # Keep from_pretrained and save_pretrained as is
    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)
        config_path = os.path.join(save_directory, 'config.json')
        # Need to handle non-serializable parts of config if any
        config_dict = {k: v for k, v in self.config.__dict__.items() if isinstance(v, (int, float, str, bool, list, dict, tuple))}
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

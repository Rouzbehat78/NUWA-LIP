# --- Executing code within /content/AiM/models/aim.py ---

import torch
import torch.nn as nn
import math # Make sure math is imported
from .stage2.config_mamba import MambaConfig
from .stage2.mixer_seq_simple import MambaLMHeadModel
# Ensure VQ_models dict and potentially VQModel class are imported
from .stage1.vq_model import VQ_models, VQModel

from huggingface_hub import PyTorchModelHubMixin

# Assume util.helper is available or instantiate_from_config is defined elsewhere if needed
# from util.helper import instantiate_from_config


class AiM(nn.Module, PyTorchModelHubMixin, repo_url="https://github.com/hp-l33/AiM", pipeline_tag="unconditional-image-generation", license="mit"):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config # Store config

        # 1. Init VQVAE FIRST
        self.vqvae = self.init_1st_stage_model()
        # print(f"Initialized VQVAE: {type(self.vqvae)}") # Optional Debug

        # 2. Determine Mamba vocab_size using the initialized VQVAE
        try:
            # Access the codebook size from the instantiated vqvae model
            if hasattr(self.vqvae, 'config') and hasattr(self.vqvae.config, 'codebook_size'):
                codebook_size = self.vqvae.config.codebook_size
            elif hasattr(self.vqvae, 'n_embed'): # Common attribute name
                 codebook_size = self.vqvae.n_embed
            elif hasattr(self.vqvae, 'quantize') and hasattr(self.vqvae.quantize, 'n_embed'): # Inside quantizer
                 codebook_size = self.vqvae.quantize.n_embed
            else: # Fallback default
                print("Warning: Could not determine VQVAE codebook size. Using default 16384.")
                codebook_size = 16384

            # Adjust MambaConfig vocab_size (+1 for unconditional token)
            expected_vocab_size = codebook_size + 1
            if not hasattr(config, 'vocab_size') or config.vocab_size != expected_vocab_size:
                print(f"Adjusting MambaConfig vocab_size from {getattr(config, 'vocab_size', 'Not Set')} to {expected_vocab_size}")
                config.vocab_size = expected_vocab_size
            # else: print(f"MambaConfig vocab_size {config.vocab_size} already set correctly.") # Optional Debug

        except Exception as e:
            print(f"Error determining vocab_size from VQVAE: {e}. Using default from config if provided.")
            if not hasattr(config, 'vocab_size'):
                 default_vq_vocab = 16384 + 1
                 print(f"Setting MambaConfig vocab_size to default: {default_vq_vocab}")
                 config.vocab_size = default_vq_vocab

        # 3. Init Mamba model using the potentially adjusted config
        self.mamba = self.init_2nd_stage_model(config)
        # print(f"Initialized Mamba: {type(self.mamba)}") # Optional Debug

        # 4. Add text_embed_proj layer (Matches CLIP Base output to Mamba input)
        clip_embed_dim = 512
        mamba_hidden_dim = config.d_model
        self.text_embed_proj = nn.Linear(clip_embed_dim, mamba_hidden_dim)
        # print(f"Added text_embed_proj: Linear({clip_embed_dim}, {mamba_hidden_dim})") # Optional Debug

        # 5. Assign other properties from config
        self.num_classes = config.num_classes if hasattr(config, 'num_classes') else 1001
        self.num_tokens = config.num_tokens if hasattr(config, 'num_tokens') else 256
        self.num_embed_dim = self.vqvae.config.embed_dim if hasattr(self.vqvae, 'config') and hasattr(self.vqvae.config, 'embed_dim') else 8

        # report number of parameters (based on all initialized modules)
        print("number of parameters: %.2fM" % (sum(p.numel() for p in self.parameters())/1e6))

    def init_1st_stage_model(self):
        if 'VQ-f16' not in VQ_models:
             raise KeyError("VQ_models dictionary does not contain 'VQ-f16'. Check stage1/vq_model.py")
        model = VQ_models['VQ-f16']()
        model.eval()
        for p in model.parameters(): p.requires_grad_(False)
        return model

    # init_2nd_stage_model now just creates the Mamba model with the final config
    def init_2nd_stage_model(self, config):
        model = MambaLMHeadModel(config)
        return model

    def get_num_params(self, non_embedding=False):
        # Return params of the Mamba module + text_embed_proj if needed, or total
        n_params = sum(p.numel() for p in self.parameters()) # Total
        # Or specific parts:
        # n_params = sum(p.numel() for p in self.mamba.parameters())
        # if hasattr(self, 'text_embed_proj'): n_params += sum(p.numel() for p in self.text_embed_proj.parameters())
        if non_embedding:
             if hasattr(self.mamba, 'backbone') and hasattr(self.mamba.backbone, 'embeddings') and hasattr(self.mamba.backbone.embeddings, 'word_embeddings'):
                 n_params -= self.mamba.backbone.embeddings.word_embeddings.weight.numel()
        return n_params

    # forward isn't used by our training loop, but keep for compatibility
    def forward(self, x, c):
        # print("Warning: AiM.forward() called") # Optional Debug
        with torch.no_grad(): # Encoding should not require gradients
             code = self.encode_to_z(x)[1] if x.ndim == 4 else x.squeeze(1)
        cond = self.encode_to_c(c) # This part needs gradients for text_embed_proj
        target = code
        logits = self.mamba(code[:, :-1], cond=cond).logits
        return logits, target # Note: slicing might be needed depending on mamba output length

    # --- sample_cfg: Corrected version ---
    @torch.no_grad()
    def sample_cfg(self, sos_token, temperature=1.0, top_k=0, top_p=1.0, fast=True, **kwargs):
        # sos_token is the conditional float embedding [B, D_mamba]
        batch_size = sos_token.shape[0]
        sos_token_cfg = torch.cat([sos_token, sos_token], dim=0) # Double cond embedding

        start_token_id = self.num_classes # Use uncond token ID
        start_input_ids = torch.full((batch_size * 2, 1), start_token_id, dtype=torch.long, device=sos_token.device)

        max_length = self.num_tokens + 1 # Total sequence length

        # Call Mamba Generate
        x = self.mamba.generate(
            input_ids=start_input_ids,    # Integer IDs [2B, 1]
            cond=sos_token_cfg,           # Float conditioning [2B, D_mamba]
            max_length=max_length,
            temperature=temperature, top_p=top_p, top_k=top_k,
            cg=fast, **kwargs
        )

        if hasattr(self.mamba, '_decoding_cache'): self.mamba._decoding_cache = None # Clear cache if exists
        tokens_no_start = x[:, 1:] # Remove start token -> [2B, num_tokens]
        return tokens_no_start[:batch_size] # Return conditional part -> [B, num_tokens]

    # --- generate: Corrected version ---
    @torch.no_grad()
    def generate(self, c=None, batch=4, temperature=1.0, top_k=0, top_p=1.0, cfg_scale=5.0, fast=True):
        if c is not None: batch = c.shape[0]
        else: print("Warning: Generating unconditionally."); pass # Handle None 'c' if needed

        self.mamba.cfg_scale = cfg_scale

        # Ensure 'c' is the projected float embedding [B, D_mamba]
        if not (isinstance(c, torch.Tensor) and c.dtype in (torch.float32, torch.float16, torch.bfloat16)):
             print("Error: generate() expects 'c' as float embedding tensor.")
             if isinstance(c, torch.Tensor) and c.dtype in (torch.int64, torch.int32):
                  print("Warning: 'c' was integer labels, calling encode_to_c (requires grad).")
                  # Temporarily enable grad for encode_to_c if needed, though generate is no_grad
                  with torch.enable_grad(): # May not be needed if proj layer doesn't require grad here
                     c = self.encode_to_c(c)
             else: raise TypeError("Unsupported 'c' type for generation.")

        # Call sample_cfg with the float embedding
        tokens = self.sample_cfg(c, temperature=temperature, top_k=top_k, top_p=top_p, fast=fast)

        if tokens is None or tokens.numel() == 0: print("Error: sample_cfg returned no tokens."); return None
        if tokens.dtype not in [torch.int64, torch.int32]: tokens = tokens.long()

        print(f"DEBUG generate(): Shape of 'tokens' passed to decode_to_img: {tokens.shape}")
        print(f"DEBUG generate(): Number of elements in 'tokens': {tokens.numel()}")

        # Decode
        T = self.num_tokens
        H = W = int(math.sqrt(T)) # Use math.sqrt
        if H * W != T: print(f"Warning: num_tokens {T} not perfect square.")
        if not hasattr(self, 'num_embed_dim'): self.num_embed_dim = self.vqvae.config.embed_dim
        shape = (batch, self.num_embed_dim, H, W)
        imgs = self.decode_to_img(tokens, shape)
        return imgs

    @torch.no_grad()
    def encode_to_z(self, x):
        # Add check for VQVAE existence
        if not hasattr(self, 'vqvae'): raise AttributeError("VQVAE model not found in AiM instance.")
        quant_z, _, log = self.vqvae.encode(x)
        # Ensure log format is as expected (list or tuple, last element is indices)
        if not isinstance(log, (list, tuple)) or not len(log): raise ValueError("Unexpected log format from vqvae.encode")
        indices = log[-1].view(quant_z.shape[0], -1)
        return quant_z, indices

    # --- encode_to_c: Corrected (NO @torch.no_grad()) ---
    # REMOVED @torch.no_grad() decorator
    def encode_to_c(self, c):
        # print("DEBUG: Inside encode_to_c...") # Optional debug
        if isinstance(c, torch.Tensor) and c.dtype in (torch.float32, torch.float16, torch.bfloat16):
            if hasattr(self, "text_embed_proj"):
                # print(f"DEBUG: encode_to_c (float path). Input 'c' shape: {c.shape}") # Optional debug
                projected_c = self.text_embed_proj(c) # Shape: [B, D_mamba]
                return projected_c
            else:
                print("⚠️ text_embed_proj layer missing! Returning raw input 'c'.")
                return c
        elif isinstance(c, torch.Tensor) and c.dtype in (torch.int64, torch.int32):
            # This path returns integer IDs, used for label conditioning if implemented elsewhere
            sos_tokens = c.contiguous().reshape(-1, 1)
            return sos_tokens.long()
        else:
            print(f"⚠️ Unexpected type for 'c' in encode_to_c: {type(c)}, dtype: {getattr(c, 'dtype', 'N/A')}")
            try:
                # Attempt conversion assuming list/tuple of ints
                device = next(self.parameters()).device # Get model device
                c_tensor = torch.tensor(c, dtype=torch.long, device=device)
                sos_tokens = c_tensor.contiguous().reshape(-1, 1)
                return sos_tokens.long()
            except Exception as e:
                print(f"Failed fallback conversion in encode_to_c: {e}")
                raise TypeError("Unsupported type passed to encode_to_c")

    @torch.no_grad()
    def decode_to_img(self, index, z_shape):
        if not hasattr(self, 'vqvae'): raise AttributeError("VQVAE model not found in AiM instance.")
        if index.dtype not in [torch.int64, torch.int32]: index = index.long()
        x = self.vqvae.decode_code(index, shape=z_shape)
        return x

# --- Factory Functions ---
def AiM_B(**kwargs):
    config_args = {'d_model': 768, 'n_layer': 24, 'adaln_group': False}
    config_args.update(kwargs)
    # vocab_size determined in __init__ now
    config_args.setdefault('num_classes', 1001)
    config_args.setdefault('num_tokens', 256)
    return AiM(MambaConfig(**config_args))

def AiM_L(**kwargs):
    config_args = {'d_model': 1024, 'n_layer': 48, 'adaln_group': True, 'num_groups': 4}
    config_args.update(kwargs)
    config_args.setdefault('num_classes', 1001)
    config_args.setdefault('num_tokens', 256)
    return AiM(MambaConfig(**config_args))

def AiM_XL(**kwargs):
    config_args = {'d_model': 1536, 'n_layer': 48, 'adaln_group': True, 'num_groups': 4, 'expand': 4.197916666666667}
    config_args.update(kwargs)
    config_args.setdefault('num_classes', 1001)
    config_args.setdefault('num_tokens', 256)
    return AiM(MambaConfig(**config_args))

AiM_models = {'AiM-B': AiM_B, 'AiM-L': AiM_L, 'AiM-XL': AiM_XL}

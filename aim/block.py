# References:
#   Mamba:  https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
#   VAR:    https://github.com/FoundationVision/VAR/blob/main/models/var.py

from typing import Optional

import torch
from torch import nn, Tensor

from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
 

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False,
        residual_in_fp32=False, adaln_group=False, mixer_drop=0.0, mlp_drop=0.0
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)

        # modify
        self.mixer_dropout = nn.Dropout(mixer_drop)
        self.adaln_group = adaln_group
        self.adaln_factor = 3   # alpha, beta, gamma

        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
            self.adaln_factor += 3
            self.mlp_dropout = nn.Dropout(0.0)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
        
        # adaLN
        if adaln_group:
            self.scale_shift_table = nn.Parameter(torch.randn(1, self.adaln_factor, dim) / dim**0.5)
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, self.adaln_factor * dim, bias=True)
            )
            # zero-out
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, cls_embed=None, inference_params=None, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required). Shape [B, S, D]
            residual: Output of residual branch from previous layer. Shape [B, S, D] or None
            cls_embed: Conditioning embedding [B, D] or None. Passed to internal AdaLN.
            inference_params: Parameters for inference caching.
        """
        # Store original input for potential return as residual if needed by fused norm logic later
        # hidden_states_input = hidden_states # Potentially needed if return signature changes
        layer_idx_str = f"Block {getattr(self, 'layer_idx', 'N/A')}" # For logging

        # --- Normalization + Potential Add ---
        # hidden_states_normed: Output of LN/RMSNorm
        # residual: Updated residual tensor. For fused_add_norm=True, it's hidden_states + input_residual.
        #           For fused_add_norm=False, it's the input residual (or None).
        if not self.fused_add_norm:
            # Apply residual connection first if it exists
            residual_input_norm = (hidden_states + residual) if residual is not None else hidden_states
            # Apply normalization
            hidden_states_normed = self.norm(residual_input_norm.to(dtype=self.norm.weight.dtype))
            # Keep the input residual state for the MLP path if it exists
            residual_for_mlp = residual_input_norm
            if self.residual_in_fp32 and residual_for_mlp is not None:
                residual = residual_for_mlp.to(torch.float32) # Preserve precision if needed for next block add
            else:
                residual = residual_for_mlp # Use the result of the add for next block if not fp32

        else:
            # Use fused add+norm
            if layer_norm_fn is None: # Check if kernel is available
                 print(f"Warning: {layer_idx_str}: Fused norm disabled (Triton unavailable). Using non-fused path fallback.")
                 residual_input_norm = (hidden_states + residual) if residual is not None else hidden_states
                 hidden_states_normed = self.norm(residual_input_norm.to(dtype=self.norm.weight.dtype))
                 # residual for next block is the state after add
                 residual = residual_input_norm
                 if self.residual_in_fp32 and residual is not None: residual = residual.to(torch.float32)
            else:
                 # residual passed in here is the *input* residual from the previous block
                 hidden_states_normed, residual = layer_norm_fn( # 'residual' is updated by layer_norm_fn!
                     hidden_states,
                     self.norm.weight,
                     self.norm.bias,
                     residual=residual,
                     prenorm=True, # Apply norm *before* mixer/mlp
                     residual_in_fp32=self.residual_in_fp32,
                     eps=self.norm.eps,
                     is_rms_norm=isinstance(self.norm, RMSNorm)
                 )
                 # 'residual' now holds the result of hidden_states + input_residual
        # --- End Normalization ---


        # --- AdaLN Parameter Calculation ---
        shift_mixer, scale_mixer, gate_mixer = None, None, None
        shift_mlp, scale_mlp, gate_mlp = None, None, None
        # Determine factor based on whether MLP exists *in this instance*
        current_adaln_factor = 6 if self.mlp is not None else 3

        if cls_embed is None:
            # print(f"{layer_idx_str}: cls_embed is None. Using zero modulation.") # Debug
            _B, _S, _D = hidden_states_normed.shape # Get shape from normed state
            shift_mixer = torch.zeros((_B, _D), device=hidden_states_normed.device, dtype=hidden_states_normed.dtype)
            scale_mixer = torch.zeros((_B, _D), device=hidden_states_normed.device, dtype=hidden_states_normed.dtype)
            gate_mixer = torch.ones((_B, _D), device=hidden_states_normed.device, dtype=hidden_states_normed.dtype) # Gate=1 -> pass through
            if self.mlp is not None:
                 shift_mlp = torch.zeros((_B, _D), device=hidden_states_normed.device, dtype=hidden_states_normed.dtype)
                 scale_mlp = torch.zeros((_B, _D), device=hidden_states_normed.device, dtype=hidden_states_normed.dtype)
                 gate_mlp = torch.ones((_B, _D), device=hidden_states_normed.device, dtype=hidden_states_normed.dtype)

        elif self.adaln_group:
             # This path is likely NOT taken for AiM-B/XL config
             print(f"{layer_idx_str}: ERROR - Unexpectedly in adaln_group path!")
             # Logic here assumes cls_embed is broadcastable with scale_shift_table
             scale_shift_params = (self.scale_shift_table + cls_embed).unbind(1)
             if current_adaln_factor == 3: shift_mixer, scale_mixer, gate_mixer = scale_shift_params
             elif current_adaln_factor == 6: shift_mixer, shift_mlp, scale_mixer, scale_mlp, gate_mixer, gate_mlp = scale_shift_params
             else: raise ValueError(f"Unsupported adaln_factor {current_adaln_factor} with adaln_group")

        else:
            # Standard AdaLN path (used for AiM-B/XL)
            if not hasattr(self, 'adaLN_modulation'):
                 raise AttributeError(f"{layer_idx_str}: ERROR - adaLN_modulation layer missing!")
            # cls_embed shape: [B, D] -> adaLN_modulation outputs [B, factor*D]
            scale_shift_params_all = self.adaLN_modulation(cls_embed)
            scale_shift_params = scale_shift_params_all.chunk(current_adaln_factor, dim=1) # Each chunk [B, D]

            if current_adaln_factor == 3:
                shift_mixer, scale_mixer, gate_mixer = scale_shift_params
            elif current_adaln_factor == 6:
                shift_mixer, shift_mlp, scale_mixer, scale_mlp, gate_mixer, gate_mlp = scale_shift_params
            else: # Should not happen
                raise ValueError(f"Unsupported adaln_factor value {current_adaln_factor}")

        # Ensure params are assigned
        if shift_mixer is None or scale_mixer is None or gate_mixer is None:
             raise ValueError(f"{layer_idx_str}: AdaLN mixer parameters not assigned correctly.")
        if self.mlp is not None and (shift_mlp is None or scale_mlp is None or gate_mlp is None):
             raise ValueError(f"{layer_idx_str}: AdaLN MLP parameters not assigned correctly.")
        # --- End AdaLN ---


        # --- Mixer Path ---
        # Print shapes RIGHT BEFORE the problematic operation
        # print(f"\n{layer_idx_str} BEFORE Mixer Modulate:")
        # print(f"  hidden_states_normed shape: {hidden_states_normed.shape}")
        # print(f"  shift_mixer shape:          {shift_mixer.shape}")
        # print(f"  scale_mixer shape:          {scale_mixer.shape}")
        try:
            modulated_mixer_input = modulate(hidden_states_normed, shift_mixer, scale_mixer)
        except RuntimeError as e:
             print(f"!!! RUNTIME ERROR during Mixer modulate() call in {layer_idx_str} !!!")
             raise e # Re-raise the error after printing shapes

        mixer_out = self.mixer(modulated_mixer_input, inference_params=inference_params, **mixer_kwargs)

        # Print shapes RIGHT BEFORE the problematic operation
        # print(f"\n{layer_idx_str} BEFORE Mixer Gating:")
        # print(f"  gate_mixer shape:  {gate_mixer.shape}")
        # print(f"  mixer_out shape:   {mixer_out.shape}")
        try:
             hidden_states_after_mixer = self.mixer_dropout(gate_mixer.unsqueeze(1) * mixer_out)
        except RuntimeError as e:
             print(f"!!! RUNTIME ERROR during Mixer Gating * call in {layer_idx_str} !!!")
             raise e # Re-raise the error after printing shapes

        # Output of this stage
        current_hidden_states = hidden_states_after_mixer

        # --- MLP Path (if applicable) ---
        if self.mlp is not None:
             # Get the correct residual state *before* the MLP's normalization step
             if not self.fused_add_norm:
                 # residual_for_mlp is the output of the mixer path added to the residual *before* the mixer's norm
                 residual_for_mlp = current_hidden_states + residual
                 hidden_states_normed_for_mlp = self.norm2(residual_for_mlp.to(dtype=self.norm2.weight.dtype))
                 # Update the residual that will be passed to the *next* block
                 residual = residual_for_mlp # State before norm2
                 if self.residual_in_fp32: residual = residual.to(torch.float32)

             else:
                 # fused_add_norm: residual already contains hidden_states_input + input_residual
                 # We need to add the mixer output to this *before* norm2
                 residual_for_mlp = current_hidden_states + residual # Add mixer output
                 # Apply fused norm2
                 if layer_norm_fn is None: # Fallback if Triton unavailable
                     print(f"Warning: Fused norm MLP path fallback in Block {layer_idx_str}")
                     hidden_states_normed_for_mlp = self.norm2(residual_for_mlp.to(dtype=self.norm2.weight.dtype))
                     residual = residual_for_mlp # Update residual for next block
                     if self.residual_in_fp32: residual = residual.to(torch.float32)
                 else:
                     hidden_states_normed_for_mlp, residual = layer_norm_fn( # residual is updated here
                         current_hidden_states, # Input to norm is mixer output
                         self.norm2.weight, self.norm2.bias,
                         residual=residual, # Pass residual *before* adding mixer output
                         prenorm=True, residual_in_fp32=self.residual_in_fp32,
                         eps=self.norm2.eps, is_rms_norm=isinstance(self.norm2, RMSNorm)
                     )
                 # 'residual' now holds the state just before MLP norm (output of fused add+norm)

             # Modulate the normalized input for MLP
            #  print(f"\n{layer_idx_str} BEFORE MLP Modulate:")
            #  print(f"  hidden_states_normed_for_mlp shape: {hidden_states_normed_for_mlp.shape}")
            #  print(f"  shift_mlp shape:                    {shift_mlp.shape}")
            #  print(f"  scale_mlp shape:                    {scale_mlp.shape}")
             try:
                 modulated_mlp_input = modulate(hidden_states_normed_for_mlp, shift_mlp, scale_mlp)
             except RuntimeError as e:
                 print(f"!!! RUNTIME ERROR during MLP modulate() call in {layer_idx_str} !!!")
                 raise e

             mlp_out = self.mlp(modulated_mlp_input)

             # Apply gating and dropout
            #  print(f"\n{layer_idx_str} BEFORE MLP Gating:")
            #  print(f"  gate_mlp shape: {gate_mlp.shape}")
            #  print(f"  mlp_out shape:  {mlp_out.shape}")
             try:
                 current_hidden_states = self.mlp_dropout(gate_mlp.unsqueeze(1) * mlp_out)
             except RuntimeError as e:
                 print(f"!!! RUNTIME ERROR during MLP Gating * call in {layer_idx_str} !!!")
                 raise e

             # Update final output
             final_hidden_states = current_hidden_states

        else: # No MLP
             final_hidden_states = current_hidden_states
             # Update residual for next block if necessary (depends on fused_add_norm path)
             if not self.fused_add_norm:
                  residual = residual_input_norm # Use state before mixer norm

        # --- Return ---
        # Return the final output of the block's main path,
        # and the residual state to be added *before* the *next* block's normalization.
        return final_hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

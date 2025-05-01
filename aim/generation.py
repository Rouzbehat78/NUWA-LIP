# References:
#   Mamba:  https://github.com/state-spaces/mamba/blob/main/mamba_ssm/utils/generation.py

import gc
import time
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F

# Add import for einops if used
try:
    from einops import rearrange, repeat
except ImportError:
    rearrange = None
    repeat = None
    print("Warning: 'einops' not installed. Rearrange operations might fail if used by model.")

from torch import Tensor
from torch.profiler import ProfilerActivity, profile, record_function

# Use GenerationOutputs if available, otherwise define namedtuples
try:
    from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput, TextStreamer
except ImportError:
    print("Warning: `transformers.generation` not found. Defining dummy output classes.")
    GreedySearchDecoderOnlyOutput = namedtuple("GreedySearchDecoderOnlyOutput", ["sequences", "scores"])
    SampleDecoderOnlyOutput = namedtuple("SampleDecoderOnlyOutput", ["sequences", "scores"])
    TextStreamer = None # Define as None if not available


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""
    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()

# --- Logit Modification Functions ---
def modify_logits_for_min_p_filtering(logits, min_p):
    """Set the logits for none min_p values to -inf. Done in-place."""
    if min_p is None: return
    # Ensure min_p is a tensor for comparison
    if not isinstance(min_p, torch.Tensor):
         min_p = torch.tensor(min_p, device=logits.device, dtype=logits.dtype)
    if (min_p <= 0.0).any() or (min_p >= 1.0).any(): return
    indices_to_remove = logits < min_p; logits.masked_fill_(indices_to_remove, float("-Inf"))

def modify_logits_for_top_k_filtering(logits, top_k):
    """Set the logits for none top-k values to -inf. Done in-place."""
    if top_k is None or top_k <= 0: return
    k = min(top_k, logits.size(-1)); indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]; logits.masked_fill_(indices_to_remove, float("-Inf"))

def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for none top-p values to -inf. Done in-place."""
    if top_p is None or top_p <= 0.0 or top_p >= 1.0: return
    sorted_logits, sorted_indices = torch.sort(logits, descending=True) # Sort descending
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone() # Keep first token that exceeds p
    sorted_indices_to_remove[..., 0] = 0 # Always keep the most probable token
    indices_to_remove = sorted_indices_to_remove.scatter(logits.dim() - 1, sorted_indices, sorted_indices_to_remove); logits.masked_fill_(indices_to_remove, float("-inf"))

def modify_logit_for_repetition_penalty(logits, prev_output_tokens, repetition_penalty=1.0):
    """Apply repetition penalty."""
    if repetition_penalty == 1.0 or prev_output_tokens is None or prev_output_tokens.numel() == 0: return logits
    score = torch.gather(logits, 1, prev_output_tokens); score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
    logits.scatter_(1, prev_output_tokens, score); return logits

# --- Sampling Function ---
def sample(logits, top_k=1, top_p=0.0, min_p=0.0, temperature=1.0):
    """Sample from logits, applying temperature, top-k, top-p, min_p filtering."""
    if temperature == 0 or top_k == 1: # Greedy decoding shortcut
        return logits.argmax(dim=-1)

    logits = logits / max(temperature, 1e-8) # Apply temperature

    # Apply filters (in-place)
    if top_k > 0: modify_logits_for_top_k_filtering(logits, top_k)
    if top_p > 0.0: modify_logits_for_top_p_filtering(logits, top_p)
    if min_p > 0.0:
        probs_after_filter = torch.softmax(logits, dim=-1)
        max_prob = probs_after_filter.max(dim=-1, keepdim=True).values
        min_p_thresh = max_prob * min_p
        modify_logits_for_min_p_filtering(logits, min_p_thresh)

    # Sample from the filtered distribution
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0) # Handle potential NaNs after filtering
    # Renormalize if needed (optional, multinomial handles unnormalized)
    # probs_sum = probs.sum(dim=-1, keepdim=True)
    # probs = torch.where(probs_sum > 1e-6, probs / probs_sum, torch.zeros_like(probs) + 1/probs.shape[-1])

    return torch.multinomial(probs, num_samples=1).squeeze(-1) # Output shape [batch_size]

# --- get_logits function (cache bypassed version) ---
# Needs 'model' passed to it (e.g., via partial or direct arg)
def get_logits(input_ids, inference_params, model, cond=None):
    """
    Calculates logits for the next token prediction, bypassing decoding cache.
    """
    # Determine position_ids based on decoding step
    decoding = inference_params.seqlen_offset > 0
    current_batch_size = input_ids.shape[0] # Batch size might be B or 2B

    if decoding:
        # For subsequent decoding steps, position is the current offset
        position_ids = torch.full(
            (current_batch_size, 1), # Shape [B or 2B, 1]
            inference_params.seqlen_offset,
            dtype=torch.long,
            device=input_ids.device,
        )
    else:
        # For the initial prompt processing step (seqlen_offset == 0)
        position_ids = None # Let the model handle positional embeddings for the sequence

    # --- Always call the model directly, bypassing cache ---
    # print(f"get_logits: Calling model directly. Input shape: {input_ids.shape}, Cond shape: {cond.shape if cond is not None else 'None'}") # Debug
    logits = model( # Call MambaLMHeadModel.forward
        input_ids=input_ids,
        position_ids=position_ids,
        cond=cond, # Pass conditioning information
        inference_params=inference_params, # Pass inference state
        num_last_tokens=1 # We only need the logits for the very last token position
    ).logits # Extract logits from output tuple

    # Output logits shape is likely [B or 2B, SeqLen, VocabSize]
    # We need the logits for the last token: [B or 2B, VocabSize]
    if logits.ndim == 3 and logits.shape[1] > 0:
         logits = logits[:, -1, :] # Select logits for the last token in the sequence
    elif logits.ndim == 2:
         # If output is already [B or 2B, Vocab], assume it's correct
         pass
    else:
         print(f"Warning: Unexpected logits shape from model: {logits.shape}")

    return logits # Return shape [B or 2B, VocabSize]
# --- End get_logits function ---


# --- should_stop function ---
def should_stop(current_token_seq, inference_params, max_length, eos_token_id=None):
    """Determines if generation should stop."""
    # current_token_seq shape: [B or 2B, 1]
    if inference_params.seqlen_offset == 0: # Don't stop right away
        return False
    # Check for EOS token in the last generated tokens
    if eos_token_id is not None and (current_token_seq[:, -1] == eos_token_id).all():
        return True
    # Check for max length (seqlen_offset is length *before* current token)
    if inference_params.seqlen_offset >= max_length - 1:
        return True
    return False
# --- End should_stop function ---


# --- decode function (main generation loop) ---
@torch.inference_mode()
def decode(
    input_ids, # Initial input IDs, shape [B, seqlen_og] (usually seqlen_og=1)
    model, # The MambaLMHeadModel instance
    max_length, # The maximum *total* sequence length
    top_k=1, top_p=0.0, min_p=0.0, temperature=1.0,
    repetition_penalty=1.0, eos_token_id=None, teacher_outputs=None, # teacher_outputs unused
    cg=False, # Classifier-Free Guidance flag
    enable_timing=False,
    cond=None, # Conditioning tensor (float embeddings [B, D_model] or None)
    streamer: Optional[TextStreamer] = None,
    **kwargs # Catch extra args
):
    """Autoregressive decoding loop - Simplified, Cache Bypassed for CFG"""

    # --- Setup ---
    original_batch_size, seqlen_og = input_ids.shape
    vocab_size = model.config.vocab_size if hasattr(model, 'config') and hasattr(model.config, 'vocab_size') else None
    if vocab_size is None: print("Warning: Cannot determine vocab_size from model config.")
    effective_batch_size = original_batch_size * 2 if cg else original_batch_size

    # --- Create Basic InferenceParams MANUALLY ---
    # No caching involved here. Primarily used to track seqlen_offset.
    inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=effective_batch_size)
    inference_params.key_value_memory_dict = {} # Mamba likely doesn't use this externally
    # Initialize lengths per sample if needed by model internals (unlikely for Mamba)
    # inference_params.lengths_per_sample = torch.full((effective_batch_size,), seqlen_og, dtype=torch.int32, device=input_ids.device)
    inference_params.seqlen_offset = 0 # Start at offset 0

    # --- Handle CFG Input Doubling (First Step Only) ---
    if cg:
        input_ids = torch.cat([input_ids, input_ids], dim=0) # [2B, S_init]
        if cond is not None:
            if cond.shape[0] == original_batch_size:
                cond = torch.cat([cond, cond], dim=0) # [2B, D]
            elif cond.shape[0] != effective_batch_size:
                raise ValueError(f"CFG Error: Initial `cond` batch size ({cond.shape[0]}) does not match expected doubled size ({effective_batch_size}).")
        # print(f"decode: Initial CFG call. Doubled inputs. Effective B={effective_batch_size}") # Debug

    # --- Generation Loop ---
    sequences = [input_ids] # Store list of Tensors, each [B or 2B, S_step]
    scores = []             # Store raw logits output at each step [B or 2B, V]
    get_logits_fn = partial(get_logits, model=model) # Pass model instance
    current_cond = cond     # Conditioning used in the loop (potentially doubled)

    if enable_timing: start_event = torch.cuda.Event(enable_timing=True); end_event = torch.cuda.Event(enable_timing=True); start_event.record()

    # --- Loop starts here ---
    while True:
        # --- Check Stopping Condition FIRST ---
        # Check based on the length *before* generating the next token
        # Use original_batch_size view for checking EOS if using CFG
        effective_last_token = sequences[-1][:original_batch_size, -1:] if cg else sequences[-1][:, -1:]
        # Pass the LATEST offset here
        current_total_len = sum(s.shape[1] for s in sequences)
        inference_params.seqlen_offset = current_total_len - sequences[-1].shape[1] # Offset *before* this step
        if should_stop(effective_last_token, inference_params, max_length, eos_token_id):
            # print(f"Stop condition met: offset {inference_params.seqlen_offset}, max_len {max_length}") # Debug
            break

        # --- Prepare inputs for the current step ---
        last_token_seq = sequences[-1] # Get the most recent token(s) added
        step_cond = current_cond # Use the full (potentially doubled) cond for model call

        # --- Get Logits (Cache explicitly bypassed in get_logits) ---
        current_logits = get_logits_fn(last_token_seq, inference_params, cond=step_cond)
        # current_logits shape: [B or 2B, V]

        # --- Apply CFG ---
        if cg:
             if current_logits.shape[0] == original_batch_size: logits_for_sampling = current_logits
             elif current_logits.shape[0] == original_batch_size * 2:
                  cond_lg, uncond_lg = torch.chunk(current_logits, 2, dim=0)
                  logits_for_sampling = uncond_lg + (cond_lg - uncond_lg) * model.cfg_scale # Shape [B, V]
             else: raise ValueError(f"Unexpected logits batch size {current_logits.shape[0]} during CFG.")
             scores.append(logits_for_sampling.clone()) # Store guided scores
        else:
             logits_for_sampling = current_logits # Shape [B, V]
             scores.append(current_logits.clone()) # Store raw scores

        # --- Sample next token ---
        if repetition_penalty != 1.0:
             sequences_cat = torch.cat(sequences, dim=1)
             hist_for_penalty = sequences_cat[:original_batch_size] if cg else sequences_cat
             current_len_for_penalty = sum(s.shape[1] for s in sequences) # Length *before* adding next token
             logits_penalized = modify_logit_for_repetition_penalty(
                 logits_for_sampling.clone(),
                 hist_for_penalty[:, :current_len_for_penalty], # Use history up to now
                 repetition_penalty
             )
             logits_for_sampling = logits_penalized

        next_token_ids = sample(logits_for_sampling, top_k=top_k, top_p=top_p, min_p=min_p, temperature=temperature)
        next_token_seq = next_token_ids.unsqueeze(-1) # Shape [B, 1]

        # --- Append and Stream ---
        if streamer is not None: streamer.put(next_token_seq.cpu())
        if cg: sequences.append(torch.cat([next_token_seq, next_token_seq], dim=0)) # Append [2B, 1]
        else: sequences.append(next_token_seq) # Append [B, 1]

        # --- Update offset FOR THE NEXT iteration's should_stop check ---
        # This wasn't strictly necessary as we calculate current_total_len anyway
        # inference_params.seqlen_offset += last_token_seq.shape[1] # Remove this - recalculate fresh each time

    # --- End Loop ---

    # --- Finalize ---
    if streamer is not None: streamer.end()
    if enable_timing: end_event.record(); torch.cuda.synchronize(); print(f"Generation time: {(start_event.elapsed_time(end_event)):.0f}ms")

    all_sequences = torch.cat(sequences, dim=1) # Concatenate all tokens

    # Return only the conditional part if CFG was used
    final_sequences = all_sequences[:original_batch_size] if cg else all_sequences
    final_scores = tuple(scores) # Contains guided scores if CFG

    output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 and top_p == 0.0 and min_p == 0.0 else SampleDecoderOnlyOutput
    return output_cls(sequences=final_sequences, scores=final_scores)

    
# --- Generation Mixin Class ---
class GenerationMixin:
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        if hasattr(self, 'backbone') and hasattr(self.backbone, 'allocate_inference_cache'):
             return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
        else: return {}

    def generate( self, input_ids, max_length, top_k=1, top_p=0.0, min_p=0.0, temperature=1.0,
        repetition_penalty=1.0, eos_token_id=None, return_dict_in_generate=False, output_scores=False,
        cond=None, streamer=None, cg=False, **kwargs, ):
        use_cg = kwargs.get('cg', cg)
        output = decode(
            input_ids=input_ids, model=self, max_length=max_length, top_k=top_k, top_p=top_p, min_p=min_p,
            temperature=temperature, repetition_penalty=repetition_penalty, eos_token_id=eos_token_id,
            cond=cond, cg=use_cg, streamer=streamer, **kwargs )
        if not output_scores: output.scores = None
        return output if return_dict_in_generate else output.sequences# --- End GenerationMixin ---


# --- Caching Functions (Keep as is, but ensure 'cond' and **kwargs are passed) ---
@dataclass
class DecodingCGCache:
    max_batch_size: int = 0; max_seqlen: int = 0; device = None; dtype = None
    callables: dict = field(default_factory=dict); mempool = None
    inference_params: Optional[InferenceParams] = None; run: Optional[Callable] = None

@torch.inference_mode()
def update_graph_cache( model, cache, batch_size, seqlen_og, max_seqlen,
    decoding_seqlens=(1,), dtype=None, n_warmups=2, cond=None, **kwargs ):
    if cache is None: cache = DecodingCGCache()
    param_example = next(iter(model.parameters())); device = param_example.device
    if dtype is None: dtype = param_example.dtype
    if ( (device, dtype) != (cache.device, cache.dtype) or batch_size > cache.max_batch_size or max_seqlen > cache.max_seqlen ):
        cache.callables = {}; cache.mempool = None; cache.inference_params = None; gc.collect()
        cache.device, cache.dtype = device, dtype; cache.max_batch_size, cache.max_seqlen = batch_size, max_seqlen
        if hasattr(model, "allocate_inference_cache"):
            inf_cache = model.allocate_inference_cache(batch_size, max_seqlen, dtype)
            lengths_per_sample = torch.full((batch_size,), seqlen_og, dtype=torch.int32, device=device)
            cache.inference_params = InferenceParams( max_seqlen=max_seqlen, max_batch_size=batch_size, seqlen_offset=seqlen_og,
                                                    key_value_memory_dict=inf_cache, lengths_per_sample=lengths_per_sample )
            if str(device) != 'cpu': # graph pool handle only for CUDA
                try: cache.mempool = torch.cuda.graphs.graph_pool_handle()
                except Exception as e: print(f"Warning: Could not get CUDA graph pool handle: {e}"); cache.mempool = None
        else:
            print("Warning: model has no allocate_inference_cache. Decoding cache disabled.")
            cache.inference_params = InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch_size, key_value_memory_dict={}) # Basic params
            return cache # Return early if cache cannot be fully allocated

    if cache.inference_params is None: # Should not happen if logic above is correct
        print("Error: InferenceParams not created in update_graph_cache.")
        return cache

    for decoding_seqlen in decoding_seqlens:
        cache_key = (batch_size, decoding_seqlen)
        if cache_key not in cache.callables:
            try:
                 cache.callables[cache_key] = capture_graph(
                     model, cache.inference_params, batch_size, max_seqlen,
                     decoding_seqlen=decoding_seqlen, mempool=cache.mempool,
                     n_warmups=n_warmups, cond=cond, **kwargs )
            except Exception as e:
                 print(f"Warning: Failed to capture CUDA graph for {cache_key}. Cache disabled. Error: {e}")
                 cache.callables[cache_key] = None # Mark as failed

    def dispatch(input_ids, position_ids, seqlen, cond=None):
        batch_size, decoding_seqlen = input_ids.shape[:2]
        cache_key = (batch_size, decoding_seqlen)
        callable_fn = cache.callables.get(cache_key, "missing") # Use "missing" sentinel

        if callable_fn is not None and callable_fn != "missing":
            # print(f"Dispatch: Cache hit for {cache_key}.") # Debug
            # Update inference params offset before calling graph
            cache.inference_params.seqlen_offset = seqlen
            return callable_fn(input_ids, position_ids, seqlen, cond=cond)
        else:
            if callable_fn == "missing": print(f"Warning: Cache key {cache_key} not found.")
            else: print(f"Warning: Graph capture failed for {cache_key}.")
            print("Using direct model call fallback.")
            # --- Fallback Logic ---
            # Manually update offset as graph replay won't do it
            cache.inference_params.seqlen_offset = seqlen
            # Call the model directly (ensure get_logits logic is robust)
            # Note: This direct call bypasses graph performance benefits
            direct_logits = get_logits(input_ids, cache.inference_params, model=model, cond=cond)
            # Squeeze might be needed depending on what get_logits returns
            # return direct_logits.squeeze(1) # Original code squeezed here
            return direct_logits # get_logits should return [B, V] now

    cache.run = dispatch
    if cache.inference_params: cache.inference_params.seqlen_offset = 0
    return cache


def capture_graph( model, inference_params, batch_size, max_seqlen, decoding_seqlen=1, mempool=None, n_warmups=2, cond=None, **kwargs ):
    device = next(iter(model.parameters())).device
    if str(device) == 'cpu':
        print("Warning: CUDA graph capture skipped on CPU.")
        # Return a lambda that calls the model directly (less efficient)
        return lambda inp_ids, pos_ids, seqlen, cond: model(inp_ids, position_ids=pos_ids, cond=cond, inference_params=inference_params, num_last_tokens=decoding_seqlen, **kwargs).logits

    input_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)
    position_ids = torch.full((batch_size, decoding_seqlen), 0, dtype=torch.long, device=device)
    seqlen_offset_og = inference_params.seqlen_offset
    inference_params.seqlen_offset = max_seqlen - decoding_seqlen
    if inference_params.lengths_per_sample is not None: inference_params.lengths_per_sample[:] = inference_params.seqlen_offset

    # Warmup
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            logits = model( input_ids, position_ids=position_ids, cond=cond, inference_params=inference_params, num_last_tokens=decoding_seqlen, **kwargs ).logits
        s.synchronize()
        if torch.distributed.is_initialized(): torch.distributed.barrier()
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        # Record the model call within the graph
        graph_logits = model( input_ids, position_ids=position_ids, cond=cond, inference_params=inference_params, num_last_tokens=decoding_seqlen, **kwargs ).logits

    def run(new_input_ids, new_position_ids, seqlen, cond=None): # Add cond to run signature
        # Update static inputs *before* replay
        if inference_params.lengths_per_sample is not None: inference_params.lengths_per_sample[:] = seqlen
        input_ids.copy_(new_input_ids)
        position_ids.copy_(new_position_ids)
        # Note: Captured graph uses the 'cond' present during capture. Cannot update cond here.
        graph.replay()
        return graph_logits.clone() # Return cloned output tensor

    inference_params.seqlen_offset = seqlen_offset_og # Restore original offset
    return run
# --- End Caching Functions ---

## Forked from oldvq_mar

import sys
from numpy.core.numeric import False_
from transformers import BertConfig, BertModel, BertTokenizer
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

sys.path.append('../..')
from config import *
from config.dfvqgan import DFVQGAN8192
from config.mps4cc import *

from inspect import isfunction

from functools import partial
from itertools import islice, cycle

from math import log2, sqrt, ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F
import clip

from einops import rearrange, repeat
from axial_positional_embedding import AxialPositionalEmbedding
# from dalle_pytorch.transformer import Transformer
from dalle_pytorch.reversible import ReversibleSequence, SequentialSequence
from dalle_pytorch.attention import Attention, SparseAttention, SparseConvCausalAttention

import megatron
from megatron import mpu


@iterable_class
class Args(BasicArgs):
    task_name, method_name, log_dir = BasicArgs.get_log_dir(__file__)
    max_train_samples = None
    max_eval_samples = 32
    max_source_len = 35
    img_size = 256
    vqvae_vocab_size = 8192

    load_nuwa = False
    adapt_load = False

    dim = 1280
    dec_depth = 24
    enc_depth = 12
    heads = 20
    dim_head = 64
    attn_dropout = 0.1
    ff_dropout = 0.1
    ignore_index = -100

    attn_types_dec = ('full', 'nearby', 'nearby', 'nearby')
    attn_types_enc = ('full')
    kernel_size = 11

    train_batch_size = 8
    eval_batch_size = 8
    learning_rate = 5e-4
    epochs = 5000
    seed = 42
    num_workers = 1
    eval_step = 1
    save_step = 1

    tk = 128  # How many top logits we consider to sample.
    sample_K = 2  # How many times we sample
    best_n = 2  # How many times we visu for sample_K.
    temperature = 1
    model_path = os.path.join(BasicArgs.root_dir, "CLIP")

    vae_patch_size = 8
    vit_patch_size = 16

    object_mask_ratio = 0

    min_vmlm_ratio = 0.5
    max_vmlm_ratio = 0.7
    min_per_mask_ratio = 0.1
    max_per_mask_ratio = 0.3
    attempt = 5

    # set_seed(seed)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.65, 1.), ratio=(1., 1.)),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.1, contrast=0.25, saturation=0.25, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    transform_prompt = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.65, 1.), ratio=(1., 1.)),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.ToTensor(),
    ])
    
    # Need a transform suitable for CLIP input (usually ImageNet norm, [0, 1] range)
    clip_transform = transforms.Compose([ # For conditioning image input to Mamba
        transforms.Resize(224), # CLIP standard size
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])



args = Args()


# helpers


def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def cast_tuple(val, depth=1):
    if isinstance(val, list):
        val = tuple(val)
    return val if isinstance(val, tuple) else (val,) * depth

def top_k(logits, tk=1):
    # num_logits = logits.shape[-1]
    # k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, tk)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def generate_nearby_mask(patch_num_row=16, kernel_size=5):
    # nearby mask generation
    effective_kernel_size = kernel_size
    padding = effective_kernel_size // 2
    mask_block = torch.ones(patch_num_row * patch_num_row, patch_num_row + padding * 2,
                            patch_num_row + padding * 2).bool()  # [2560,10,16+padding*2,16+padding*2]
    for i in range(patch_num_row):
        for j in range(patch_num_row):
            mask_block[i * patch_num_row + j][i:i + effective_kernel_size, j:j + effective_kernel_size] = False

    mask_block = mask_block[:, padding:-padding, padding:-padding].reshape(patch_num_row * patch_num_row, patch_num_row * patch_num_row)  # [2560,2560]
    return mask_block

# class NearbyAttention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0., patch_num_row=16, kernel_size=5):
#         super().__init__()
#         inner_dim = dim_head * heads
#         self.heads = heads // mpu.get_tensor_model_parallel_world_size()
#         self.scale = dim_head ** -0.5

#         self.nb_mask = generate_nearby_mask(patch_num_row=patch_num_row, kernel_size=kernel_size)
#         self.nb_mask = F.pad(self.nb_mask, (1, 0, 1, 0), value=False)
#         i, j = self.nb_mask.shape
#         self.nb_mask = self.nb_mask.view(1, 1, i, j)

#         self.q_linear = mpu.ColumnParallelLinear(
#             dim, inner_dim, bias=False, gather_output=False)
#         self.k_linear = mpu.ColumnParallelLinear(
#             dim, inner_dim, bias=False, gather_output=False)
#         self.v_linear = mpu.ColumnParallelLinear(
#             dim, inner_dim, bias=False, gather_output=False)

#         self.to_out = mpu.RowParallelLinear(
#             inner_dim, dim,
#             input_is_parallel=True
#         )
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, q, k, v, mask=None, extra_q=0, extra_k=0):
#         b, n, _, h, device = *q.shape, self.heads, q.device

#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
#                       [self.q_linear(q)[0], self.k_linear(k)[0], self.v_linear(v)[0]])

#         dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

#         mask_value = max_neg_value(dots)

#         dots.masked_fill_(self.nb_mask.to(device), mask_value)
#         if mask is not None:
#             b, i, j = mask.shape
#             mask = mask.view(b, 1, i, j)
#             dots.masked_fill_(mask, mask_value)

#         attn = dots.softmax(dim=-1)

#         out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = self.to_out(out)[0]
#         out = self.dropout(out)
#         return out

# # moved and modified from DALLE_pytorch
# class ParallelAttention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.q_linear = mpu.ColumnParallelLinear(
#             dim, inner_dim, bias=False, gather_output=False)
#         self.k_linear = mpu.ColumnParallelLinear(
#             dim, inner_dim, bias=False, gather_output=False)
#         self.v_linear = mpu.ColumnParallelLinear(
#             dim, inner_dim, bias=False, gather_output=False)

#         self.to_out = mpu.RowParallelLinear(
#             inner_dim, dim,
#             input_is_parallel=True
#         )
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, q, k, v, mask=None):
#         b, n, _, h, device = *q.shape, self.heads, q.device

#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
#                       [self.q_linear(q)[0], self.k_linear(k)[0], self.v_linear(v)[0]])

#         dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

#         mask_value = max_neg_value(dots)

#         if mask is not None:
#             b, i, j = mask.shape
#             mask = mask.view(b, 1, i, j)
#             dots.masked_fill_(mask, mask_value)

#         attn = dots.softmax(dim=-1)

#         out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = self.to_out(out)[0]
#         out = self.dropout(out)
#         return out


# class LayerScale(nn.Module):
#     def __init__(self, dim, depth):
#         super().__init__()
#         if depth <= 18:
#             init_eps = 0.1
#         elif depth > 18 and depth <= 24:
#             init_eps = 1e-5
#         else:
#             init_eps = 1e-6

#         scale = torch.zeros(1, 1, dim).fill_(init_eps)
#         self.scale = nn.Parameter(scale)

#     def forward(self, x):
#         return x * self.scale


# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn

#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)


# class GEGLU(nn.Module):
#     def forward(self, x):
#         x, gates = x.chunk(2, dim=-1)
#         return x * F.gelu(gates)


# class FeedForward(nn.Module):
#     def __init__(self, dim, dropout=0., mult=4.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, dim * mult * 2),
#             GEGLU(),
#             nn.Dropout(dropout),
#             nn.Linear(dim * mult, dim)
#         )

#     def forward(self, x):
#         return self.net(x)


# class ParallelFeedForward(nn.Module):
#     def __init__(self, dim, dropout=0., mult=4.):
#         super().__init__()
#         # bias for column / row parallel is enabled by default
#         self.dense_h_to_8h = mpu.ColumnParallelLinear(
#             dim, dim * mult * 2,
#             gather_output=False
#         )
#         self.activation = nn.Sequential(
#             GEGLU(),
#             nn.Dropout(dropout)
#         )
#         self.dense_4h_to_h = mpu.RowParallelLinear(
#             dim * mult, dim,
#             input_is_parallel=True
#         )

#     def forward(self, x):
#         out = self.dense_h_to_8h(x)[0]
#         out = self.activation(out)
#         out = self.dense_4h_to_h(out)[0]
#         return out

# class ClipTextEncoder(nn.Module):
#     def __init__(self, model_path, dim):
#         super(ClipTextEncoder, self).__init__()
#         model, _ = clip.load(model_path, device='cpu')
#         self.token_embedding = copy.deepcopy(model.token_embedding)
#         self.positional_embedding = copy.deepcopy(model.positional_embedding)
#         self.transformer = copy.deepcopy(model.transformer)
#         self.ln_final = copy.deepcopy(model.ln_final)
#         self.cond_emb = nn.Linear(512, dim)

#     def forward(self, cond):
#         cond = self.token_embedding(cond)  # [batch_size, n_ctx, d_model]
#         cond = cond + self.positional_embedding
#         cond = cond.permute(1, 0, 2)  # NLD -> LND
#         cond = self.transformer(cond)
#         cond = cond.permute(1, 0, 2)  # LND -> NLD
#         cond = self.ln_final(cond)
#         outputs = self.cond_emb(cond)  # 512 -> dim
#         return outputs

# class SingleTransformer(nn.Module):
#     def __init__(self, attention, attention_cond, ff, dim, depth):
#         super().__init__()
#         self.atten_norm = nn.LayerNorm(dim)
#         self.attention = attention
#         self.attention_scale = LayerScale(dim, depth)

#         if attention_cond is not None:
#             self.atten_norm_cond = nn.LayerNorm(dim)
#             self.attention_cond = attention_cond
#             self.attention_scale_cond = LayerScale(dim, depth)

#         self.ff_norm = nn.LayerNorm(dim)
#         self.ff = ff
#         self.ff_scale = LayerScale(dim, depth)

#     def forward(self, x, mask, cond=None, mask_cond=None):
#         # attention
#         att = self.atten_norm(x)
#         att = self.attention(att, att, att, mask)
#         att = self.attention_scale(att)
#         x = x + att

#         # attention_condition
#         if cond is not None:
#             att = self.atten_norm_cond(x)
#             att = self.attention_cond(att, cond, cond, mask_cond)
#             att = self.attention_scale_cond(att)
#             x = x + att

#         # feedforward
#         ff = self.ff_norm(x)
#         ff = self.ff(ff)
#         ff = self.ff_scale(ff)
#         ff = x + ff
#         return ff

# class Transformer(nn.Module):
#     def __init__(
#             self,
#             *,
#             dim,
#             depth,
#             cond=True,
#             heads=8,
#             dim_head=64,
#             ff_mult=4,
#             attn_dropout=0.,
#             ff_dropout=0.,
#             args=None,
#             attn_types=None,
#             patch_size=16,
#     ):
#         super().__init__()
#         self.layers = nn.ModuleList([])

#         self.args = args
#         attn_types = cast_tuple(attn_types)
#         attn_type_layer = islice(cycle(attn_types), depth)

#         for ind, attn_type in zip(range(depth), attn_type_layer):
#             if attn_type == 'full':
#                 attn_class = ParallelAttention
#             elif attn_type == 'nearby':
#                 attn_class = partial(NearbyAttention, patch_num_row=self.args.img_size // patch_size, kernel_size=self.args.kernel_size)                 
#             else:
#                 raise ValueError(f'attention type "{attn_type}" is not valid')
#             attn_cond_class = ParallelAttention
#             self.layers.append(SingleTransformer(
#                 attn_class(dim, heads=heads,
#                            dim_head=dim_head, dropout=attn_dropout),
#                 attn_cond_class(dim, heads=heads,
#                                 dim_head=dim_head, dropout=attn_dropout) if cond else None,
#                 ParallelFeedForward(dim, mult=ff_mult, dropout=ff_dropout),
#                 dim, ind + 1
#             ))

#     def forward(self, x, mask=None, cond=None, mask_cond=None):
#         for lid in range(len(self.layers)):
#             layer = self.layers[lid]
#             x = mpu.checkpoint(layer, x, mask, cond, mask_cond)
#         return x


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        # 获取visual tokenizer
        self.vae = DFVQGAN8192()

        # 冻结 visual_tokenizer的参数
        for n, p in self.vae.named_parameters():
            p.requires_grad = False

        mamba_config = MambaConfig(
            d_model=1024,
            n_layer=48,
            d_intermediate=args.dim*4,
            vocab_size=args.vqvae_vocab_size + 1,  # +1 for mask token
            num_tokens=self.image_seq_len,
            adaln_group=True,
            num_groups=4,
            clip_dim=512
        )
        dim = self.args.dim
        #self.encoder = ClipTextEncoder(model_path=os.path.join(BasicArgs.root_dir, "CLIP", "ViT-B-32.pt"), dim=dim)
        self.vae_w = self.args.img_size // self.args.vae_patch_size
        self.vae_h = self.args.img_size // self.args.vae_patch_size
        self.vit_w = self.args.img_size // self.args.vit_patch_size
        self.vit_h = self.args.img_size // self.args.vit_patch_size
        self.image_seq_len_vae = (self.args.img_size // self.args.vae_patch_size) ** 2
        self.image_seq_len_vit = (self.args.img_size // self.args.vit_patch_size) ** 2
        self.image_seq_len = (self.args.img_size // self.args.vae_patch_size) ** 2

        # to make embedding align with tensor model para#
        self.num_image_tokens = self.args.vqvae_vocab_size
        self.vae_emb = nn.Embedding.from_pretrained(copy.deepcopy(self.vae.model.quantize.embedding.weight),
                                                    freeze=False)
        self.image_emb = nn.Linear(self.vae_emb.embedding_dim, dim)
        self.patch_emb = nn.Conv2d(
            3,
            dim,
            kernel_size=self.args.vit_patch_size,
            stride=self.args.vit_patch_size,
        )

        self.image_bos_emb = nn.Parameter(torch.randn(1, dim))
        self.image_msk_emb = nn.Parameter(torch.randn(1, dim))
        #self.image_pos_emb = AxialPositionalEmbedding(dim, axial_shape=(self.image_seq_len,))
        self.image_pos_emb_vae = AxialPositionalEmbedding(dim,
                                                      axial_shape=(1, self.vae_h, self.vae_w))
        self.image_pos_emb_vit = AxialPositionalEmbedding(dim,
                                                      axial_shape=(1, self.vit_h, self.vit_w))

        # self.transformer_dec = Transformer(
        #     dim=dim,
        #     cond=True,
        #     depth=args.dec_depth,
        #     heads=args.heads,
        #     dim_head=args.dim_head,
        #     attn_dropout=args.attn_dropout,
        #     ff_dropout=args.ff_dropout,
        #     args=self.args,
        #     attn_types=self.args.attn_types_dec,
        #     patch_size=self.args.vae_patch_size,
        # )
        # self.transformer_enc = Transformer(
        #     dim=dim,
        #     cond=False,
        #     depth=args.enc_depth,
        #     heads=args.heads,
        #     dim_head=args.dim_head,
        #     attn_dropout=args.attn_dropout,
        #     ff_dropout=args.ff_dropout,
        #     args=self.args,
        #     attn_types=self.args.attn_types_enc,
        #     patch_size=self.args.vae_patch_size,
        # )
        # self.transformer_pth = Transformer(
        #     dim=dim,
        #     cond=False,
        #     depth=args.enc_depth,
        #     heads=args.heads,
        #     dim_head=args.dim_head,
        #     attn_dropout=args.attn_dropout,
        #     ff_dropout=args.ff_dropout,
        #     args=self.args,
        #     attn_types=self.args.attn_types_enc,
        #     patch_size=self.args.vit_patch_size,
        # )
        self.mamba_dec = MambaLMHeadModel(mamba_config)

        self.to_logits_img = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.num_image_tokens),
        )
        # Define mask token ID (should be outside VQGAN's vocab range)
        self.mask_token_id = args.vqvae_vocab_size
        print(f"Using mask token ID: {self.mask_token_id}")
        self.ignore_index = args.ignore_index
        self.clip_sim = CLIPSimilarity(model_filename=os.path.join(BasicArgs.root_dir, "CLIP", "ViT-B-32.pt"))
        for n, p in self.clip_sim.named_parameters():
            p.requires_grad = False
        # self.aus = nn.Linear(120, 33)
        # self.dummy_param = nn.Parameter(torch.empty(0))

    def load_state_dict(self, state):
        if self.args.load_nuwa:
            new_state = OrderedDict()
            for key in state:
                if "transformer." in key and "clip_sim." not in key and "encoder." not in key:
                    new_key = key.replace("transformer.", "transformer_dec.")
                    new_state[new_key] = state[key]
            super(Net, self).load_state_dict(new_state, strict=False)
        else:
            super(Net, self).load_state_dict(state)




    def forward(self, inputs):
        """
        Main forward function called by the training/evaluation loop.
        Dispatches to training or inference logic.
        """
        self.vae.eval() # Ensure VQGAN is in eval mode
        self.mamba_dec.eval() # Default to eval, set to train mode below if needed

        if self.training:
            self.mamba_dec.train()
            outputs = self.forward_train(inputs)
        else:
            # Inference mode
            outputs = self.forward_inference(inputs)

        return outputs

    def forward_train(self, inputs, return_loss=True, image_base=None): 
        """ Handles the forward pass during training. """
        device = self.device
        B = inputs['label_imgs'].shape[0]
        vae_cache_1 = inputs["vae_mask"].eq(1).reshape(B, -1, 1)
        vae_cache_0 = ~vae_cache_1
        # 1. Encode Image with VQGAN
        # Get ground truth VQ codes for the full image
        if image_base is None:
            image_base, _, _ = self.vae.get_codebook_indices(inputs['label_imgs'], inputs["vision_mask"])
        if image_base is None:
            gt_image_codes, gt_hm, gt_hms = self.vae.get_codebook_indices(inputs['label_imgs'], inputs["vision_mask"])
        else:
            gt_image_codes = image_base
            #_image_base, _hm, _hms = self.vae.get_codebook_indices(inputs["masked_image"], inputs["vision_mask"].bool())
            # gt_image_codes shape: [B, H*W] where H, W are latent map dimensions
        
        codes = gt_image_codes.clone()
        latent_mask = inputs["vae_mask"].reshape(B, -1) 
        codes[latent_mask] = self.mask_token_id   ## replace the masked area with mask token
        input_ids = codes  # shape (B, L), dtype=torch.long


        image = gt_image_codes.view(B, -1)

        image_target = image * vae_cache_1.view(B, -1) -100 * vae_cache_0.view(B, -1)
        image_target = image_target.long()
        if (input_ids.size(1) != self.image_seq_len):
            print("input_ids.shape[1] != self.image_seq_len")
        # Position IDs (standard range)
        position_ids = torch.arange(self.image_seq_len, device=device).unsqueeze(0).repeat(B, 1)

        # Conditioning Tuple for Mamba
        # Ensure image passed for conditioning is preprocessed for Mamba's internal CLIP
        # Use the *original* (label) image, not the masked one
        with torch.no_grad():
            cond_image_clip = self.mamba_model.backbone.clip_preprocess(inputs["masked_image"]).to(input_ids.device)

        cond = (
            inputs["input_text"],                     # list of b strings
            cond_image_clip,                   # [B,3,224,224] for clip
            inputs["vision_mask"].unsqueeze(1)        # [B,1,H,W]
        )

        # 3. Mamba Forward Pass
        # Mamba's forward expects labels for loss calculation
        mamba_output = self.mamba_model(
            input_ids=input_ids,
            position_ids=position_ids,
            cond=cond,
        )
        logits = mamba_output['logits']

        #logits_seq = self.to_logits_img(logits[:, :-1, :]).reshape(B, -1, self.num_image_tokens) not needed already output is vocab size

        outputs = {}

        if return_loss:
            #loss_img = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), image_target)
            shift_logits = logits[..., :-1, :].contiguous() # Shape: [B, L-1, VQ_Vocab_Size]
            shift_labels = image_target[..., 1:].contiguous() # Shape: [B, L-1] (Contai
            loss_img = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=self.ignore_index)
            if torch.isnan(loss_img).any():
                print("Warning: NaN loss detected!")
            outputs['loss_total'] = loss_img
        else:
            outputs['logits_seq'] = logits

        return outputs

    @torch.no_grad()
    def forward_inference(self, inputs):
        device      = self.device
        B           = inputs['masked_image'].shape[0]
        K           = self.args.sample_K
        best_n      = self.args.best_n
        T           = self.image_seq_len
        temp        = self.args.temperature
        top_k_val   = self.args.tk

        # 1) VQGAN encode both GT and masked images
        gt_codes, gt_hm, gt_hms = self.vae.get_codebook_indices(
            inputs['label_imgs'], inputs['vision_mask'].bool()
        )
        init_codes, hm, hms     = self.vae.get_codebook_indices(
            inputs['masked_image'], inputs['vision_mask'].bool()
        )
        # flatten mask: [B, L]
        flat_mask = inputs['vae_mask'].reshape(B, -1).bool()

        # 2) Expand to B*K for sampling
        #   sequences of codebook indices
        codes_k = init_codes.repeat_interleave(K, dim=0)        # [B*K, L]
        mask_k  = flat_mask.repeat_interleave(K,  dim=0)       # [B*K, L]

        # 3) Build conditioning tuple once
        #    — text
        texts = inputs['input_text']
        texts_k = [t for t in texts for _ in range(K)]          # length B*K
        #    — CLIP image preprocess
        clip_im = self.mamba_model.backbone.clip_preprocess(
            inputs['masked_image']
        ).to(device)                                            # [B, 3,224,224]
        clip_im_k = clip_im.repeat_interleave(K, dim=0)        # [B*K,3,224,224]
        #    — vision mask (for spatial conditioning)
        vm = inputs['vision_mask'].unsqueeze(1)                 # [B,1,H,W]
        vm_k = vm.repeat_interleave(K, dim=0)                   # [B*K,1,H,W]

        cond = (texts_k, clip_im_k, vm_k)

        # 4) Allocate incremental SSM cache
        cache = self.mamba_model.allocate_inference_cache(
            batch_size=B*K,
            seq_len=T,
            dtype=self.mamba_model.backbone.embeddings.word_embeddings.weight.dtype,
            device=device
        )
        inference_params = {
            "key_value_memory_dict": cache,
            "seqlen_offset": 0
        }

        # 5) Autoregressively fill every position 0..T-1
        for t in range(T):
            # one-token input at position t
            input_step = codes_k[:, t:t+1]                         # [B*K,1]
            pos_step   = torch.full((B*K,1), t, device=device)     # [B*K,1]

            out = self.mamba_model(
                input_ids        = input_step,
                position_ids     = pos_step,
                cond             = cond,
                inference_params = inference_params,
            )
            # update offset
            inference_params["seqlen_offset"] += 1

            # logits for this step
            logits = out["logits"].squeeze(1)                       # [B*K, V]
            # sample only where mask==True
            filt   = top_k(logits, top_k_val)
            probs  = F.softmax(filt / temp, dim=-1)
            samp   = torch.multinomial(probs, 1).squeeze(-1)       # [B*K]

            # replace codes at t for masked positions
            codes_k[:, t] = torch.where(mask_k[:, t], samp, codes_k[:, t])

        # 6) Decode all B*K sequences at once
        final_imgs = self.vae.decode(codes_k, hm, hms)         # [B*K, C, H, W]
        final_imgs = final_imgs.view(B, K, *final_imgs.shape[1:])  # [B, K, C, H, W]

        final_seqs = codes_k.view(B, K, T)                         # [B, K, L]

        # 7) CLIP-based ranking
        flat_imgs = final_imgs.view(B*K, *final_imgs.shape[-3:])   # [(B*K),C,H,W]
        sim = self.clip_sim(texts_k, flat_imgs, batch_size=B)      # [B*K] or [B, K]
        sim = sim.view(B, K)
        sim_sorted, idx = sim.sort(dim=1, descending=True)
        best_idx = idx[:, :best_n]                                 # [B, best_n]

        # gather best_n samples
        # images: [B, best_n, C, H, W]
        best_imgs = torch.gather(
            final_imgs, 1,
            best_idx.view(B, best_n, 1, 1, 1)
                .expand(-1, -1, *final_imgs.shape[-3:])
        )
        # sequences: [B, best_n, L]
        best_seqs = torch.gather(
            final_seqs, 1,
            best_idx.view(B, best_n, 1)
                .expand(-1, -1, T)
        )

        # 8) GT decode + metrics
        gt_imgs = self.vae.decode(gt_codes, gt_hm, gt_hms)          # [B, C, H, W]
        outputs = {
            "logits_gt_vae_image" : gt_imgs.unsqueeze(1),           # [B,1,C,H,W]
            "logits_img"         : best_imgs,                      # [B,best_n,C,H,W]
            "logits_seq"         : best_seqs,                      # [B,best_n,L]
            "metric_clip"        : sim_sorted[:, 0].mean(),        # avg top-1
            "logits_text"        : inputs["input_text"],
        }
        if "mask_class" in inputs:
            outputs["logits_label_class"] = inputs["mask_class"]

        # GT CLIP score
        sim_gt = self.clip_sim(inputs["input_text"], inputs["label_imgs"], batch_size=B)
        # if sim_gt is [B,L] or [B,1], take mean per batch:
        if sim_gt.ndim>1:
            sim_gt = sim_gt.mean(dim=tuple(range(1, sim_gt.ndim)))
        outputs["metric_clip_gt"]    = sim_gt.mean()
        outputs["metric_relative"]   = outputs["metric_clip"] / (outputs["metric_clip_gt"] + 1e-6)
        outputs["loss_total"]        = torch.tensor(0.0, device=device)

        return outputs

    def forward(self, inputs):
        #torch.cuda.empty_cache()
        self.vae.eval()
        device = self.vae_emb.weight.device  # TODO Never use dummy_param, bugs for unused params of dddp.
        image = inputs.get('label_imgs', None)
        if self.training:
            outputs = self.forward_(inputs, return_loss=True)

        if not self.training:
            temperature = self.args.temperature
            image_seq_len = self.image_seq_len
            outputs = {}

            gen_images = []
            gen_seq = []
            #zero_mask = torch.zeros_like(inputs["vision_mask"]).bool()
            gt_image_base, gt_hm, gt_hms = self.vae.get_codebook_indices(inputs["label_imgs"], inputs["vision_mask"].bool())
            _image_base, _hm, _hms = self.vae.get_codebook_indices(inputs["masked_image"], inputs["vision_mask"].bool())
            for try_it in range(self.args.sample_K):
                if not getattr(args, 'do_train', False):
                    # TODO reset seed to let the model generate more samples.
                    seed = torch.seed()
                    os.environ['PYHTONHASHSEED'] = str(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)

                # TODO
                text_mask, vae_mask = inputs["text_mask"].clone(), inputs["vae_mask"].clone()
                text, image_ = inputs["input_ids"].clone(), inputs["masked_image"].clone()
                masked_image = inputs["masked_image_blackmask"].clone()
                image_base = _image_base.clone()
                B, W, H = vae_mask.shape
                B = text.size(0)
                image_base = image_base * ~vae_mask.bool().reshape(B, -1)
                # video = []
                for cur_len in tqdm(range(image_seq_len), miniters=5):
                    flatten_vae_mask = vae_mask.reshape(B, -1)
                    use_pred = flatten_vae_mask[:, cur_len].bool()
                    if use_pred.max().item():
                        new_inputs = {
                            'input_ids': text, 'label_imgs': image_,
                            "text_mask": text_mask, "vae_mask": vae_mask,
                            "masked_image": masked_image,
                        }
                        cur_outputs = self.forward_(new_inputs, return_loss=False, image_base=image_base)

                        logits = cur_outputs["logits_seq"][:,cur_len].reshape(B, self.num_image_tokens)
                        filtered_logits = top_k(logits, tk=self.args.tk)
                        probs = F.softmax(filtered_logits / temperature, dim=-1)
                        sample = torch.multinomial(probs, 1)
                        image_base[:, cur_len] = image_base[:, cur_len] * ~use_pred + sample * use_pred
                    
                    #flatten_vae_mask[:, cur_len] = 0
                    #vae_mask = flatten_vae_mask.reshape(B, W, H)

                img_seq = image_base
                images = self.vae.decode(img_seq, _hm, _hms)

                gen_images.append(rearrange(images, '(b l) c w h -> b l c w h', b=B))
                gen_seq.append(rearrange(img_seq, '(b l) d -> b l d', b=B))
            
            gt_images = self.vae.decode(gt_image_base, gt_hm, gt_hms)
            gen_gt_images = [rearrange(gt_images, '(b l) c w h -> b l c w h', b=B)]
            outputs["logits_gt_vae_image"] = torch.stack(gen_gt_images, dim=1)

            outputs["loss_total"] = torch.Tensor([0]).to(device)
            outputs['logits_img'] = torch.stack(gen_images, dim=1)
            outputs['logits_seq'] = torch.cat(gen_seq, dim=1)

            best_n = getattr(args, 'best_n', 1)
            b, k, l, c, w, h = outputs['logits_img'].size()
            my_text = inputs['input_text']
            my_image = rearrange(outputs['logits_img'], 'b k l c w h->(b k l) c w h')
            sim_matrix = self.clip_sim(my_text, my_image, batch_size=b)  # [b, 1, kl]
            sim_matrix_arrange = rearrange(sim_matrix, 'b () (k l)-> b k l', k=k).mean(axis=2)  # [b, k]

            sim_matrix_sorted, idx = torch.sort(sim_matrix_arrange, dim=1, descending=True)
            best_idx = idx[:, :best_n]
            best_imgs = outputs['logits_img'].view(b * k, l, c, w, h)[best_idx.view(b * k)].view(b, k, l, c, w, h)
            best_seqs = outputs['logits_seq'].view(b * k, l, -1)[best_idx.view(b * k)].view(b, k, l, -1)
            # Override logits_img and logits_seq to sorted version.
            outputs['logits_img'] = best_imgs  # [b, k, l, c, w, h]
            outputs['logits_seq'] = best_seqs  # [b, k, l, d]
            outputs['metric_clip'] = sim_matrix_sorted[:, 0].mean()
            outputs['logits_text'] = my_text
            outputs['logits_label_class'] = inputs["mask_class"]

            my_gt_image = inputs['label_imgs']
            sim_matrix_gt = self.clip_sim(my_text, my_gt_image, batch_size=b)  # [b, 1, l]
            sim_matrix_gt_arrange = rearrange(sim_matrix_gt, 'b () l-> b l', l=l).mean(axis=1)  # [b]
            outputs['metric_clip_gt'] = sim_matrix_gt_arrange.mean()
            outputs['metric_relative'] = outputs['metric_clip'] / outputs['metric_clip_gt']

        return outputs

    def forward_(self, inputs, return_loss=True, image_base=None):
        text = inputs['input_ids']

        device = text.device
        B = text.size(0)

        vae_cache_1 = inputs["vae_mask"].eq(1).reshape(B, -1, 1)
        vae_cache_0 = ~vae_cache_1

        tokens_clip = self.encoder(text).to(device)

        if image_base is None:
            image_base, _, _ = self.vae.get_codebook_indices(inputs['label_imgs'], inputs["vision_mask"])
        raw_image = inputs["masked_image"]
        image = image_base.view(B, -1)

        image_target = image * vae_cache_1.view(B, -1) -100 * vae_cache_0.view(B, -1)
        image_target = image_target.long()

        target_ids = self.vae_emb(image)
        image_emb = self.image_emb(target_ids).view(B, -1, self.args.dim)
        image_emb += self.image_pos_emb_vae(image_emb)

        raw_emb = self.patch_emb(raw_image).view(B, -1, self.args.dim)
        raw_emb += self.image_pos_emb_vit(raw_emb)

        tokens_dec = image_emb * vae_cache_1 + self.image_msk_emb * vae_cache_0
        tokens_dec = torch.cat((repeat(self.image_bos_emb, 'n d -> b n d', b=B), tokens_dec), dim=1)

        tokens_enc = image_emb * vae_cache_0 + self.image_msk_emb * vae_cache_1
        tokens_pth = raw_emb

        dec_len = tokens_dec.shape[1]
        enc_len = tokens_enc.shape[1]
        pth_len = tokens_pth.shape[1]
        txt_len = tokens_clip.shape[1]

        task_mask = vae_cache_1.view(B, -1)
        mask_enc_self = None#task_mask.view(B, 1, -1)
        mask_enc_pth_self = None#torch.zeros((B, 1, pth_len), dtype=torch.bool).to(device)
        
        vis_causal = torch.ones(dec_len, dec_len, device=device, dtype=torch.bool).triu_(dec_len - dec_len + 1)
        mask_dec_self = vis_causal.view(1, dec_len, dec_len)
        #mask_dec_cross_img = torch.zeros(B, 1, enc_len + pth_len, device=device).bool()
        #mask_dec_cross_txt = text.eq(0).view(B, 1, -1)
        mask_dec_cross = None#torch.cat((mask_dec_cross_img, mask_dec_cross_txt), dim=-1)
        #mask_dec_cross = torch.cat((mask_dec_cross_img, mask_dec_cross_txt), dim=-1)

        enc_out = self.transformer_enc(tokens_enc, mask=mask_enc_self)
        enc_out_pth = self.transformer_pth(tokens_pth, mask=mask_enc_pth_self)
        tokens_cond = torch.cat((enc_out, enc_out_pth, tokens_clip), dim=-2)
        #tokens_cond = torch.cat((enc_out, tokens_clip), dim=-2)

        dec_out = self.transformer_dec(tokens_dec, mask=mask_dec_self, cond=tokens_cond, mask_cond=mask_dec_cross)
        logits_seq = self.to_logits_img(dec_out[:, :-1, :]).reshape(B, -1, self.num_image_tokens)

        outputs = {}

        if return_loss:
            loss_img = F.cross_entropy(rearrange(logits_seq, 'b n c -> b c n'), image_target)

            outputs['loss_total'] = loss_img
        else:
            outputs['logits_seq'] = logits_seq

        return outputs


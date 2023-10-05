import clip
import os
from torch import nn
import numpy as np
import torch
from enum import Enum
from tqdm import tqdm, trange
import climage
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
import skimage.io as io
import PIL.Image


# class ClipDataset(Dataset):

#params:
use_beam_search = False #@param {type:"boolean"}
img_path = "/ssd_scratch/cvit/manu/clip_cap/val2014/COCO_val2014_000000290839.jpg"
model_path = "/ssd_scratch/cvit/manu/clip_cap_manu/checkpoints/coco_prefix-009.pt"
prefix_dim = 1024
prefix_length = 10
prefix_length_clip = 10
num_layers = 8

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

prefix_length = 10 


#load model
clip_model, preprocess = clip.load("RN50", device=device, jit=False) #colab
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

#dataloader/dataset
image = io.imread(img_path)
pil_image = PIL.Image.fromarray(image)
image = preprocess(pil_image).unsqueeze(0).to(device)

#MODEL 
#@title Model


class MlpTransformer(nn.Module):
     def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
         super().__init__()
         out_d = out_d if out_d is not None else in_dim
         self.fc1 = nn.Linear(in_dim, h_dim)
         self.act = act
         self.fc2 = nn.Linear(h_dim, out_d)
         self.dropout = nn.Dropout(dropout)

     def forward(self, x):
         x = self.fc1(x)
         x = self.act(x)
         x = self.dropout(x)
         x = self.fc2(x)
         x = self.dropout(x)
         return x

class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)



class Model(nn.Module):

    def __init__(self, clip_dim,prefix_len, const_len,num_layers,only_projection = False):
        super().__init__()
        self.clip_dim = clip_dim
        self.prefix_len = prefix_len
        self.const_len = const_len
        self.only_projection = only_projection
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_dim = self.gpt.transformer.wte.weight.shape[1]     #token embedding weight.shape
        self.linear = nn.Linear(self.clip_dim,self.prefix_len*(self.gpt_dim))
        self.learnable_const = nn.Parameter(torch.randn(self.const_len, self.gpt_dim), requires_grad=True)
        self.transformer = Transformer(self.gpt_dim, 8, num_layers)  #token_embedding, attn_heads, num_blocks
        
    def forward(self, clip_embed, tokens = None, mask = None):
        #prefix --> linear layer --> transformer --> output + caption tokens --> gpt        
        
        x = self.linear(clip_embed.to(torch.float32)).view(clip_embed.shape[0],self.prefix_len, -1) # (B,K,gpt_dim)
        #concat learnable constant and clip embedding mapped to gpt space
        learnable_const = self.learnable_const.unsqueeze(0).expand(clip_embed.shape[0],*self.learnable_const.shape) # (B,K,gpt_dim)

        x = torch.cat((x,learnable_const),dim = 1) # (B,2K,gpt_dim)
        # improve the mapping in the gpt space using a transformer. Also encode image info in learnable constant through attention.
        x = self.transformer(x)[:,self.prefix_len:]
        
        if self.only_projection:
            return x 
        
        # feed the gpt (learnable constant + tokenized_caption)
        token_embed = self.gpt.transformer.wte(tokens)    # (vocab_size, token_embedding=768) --> (B,T, 768)        
        x = torch.cat((x,token_embed),dim = 1)

        out = self.gpt(inputs_embeds = x, attention_mask = mask)

        return out


#main predict class with forward
 


def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device # (10,768)

    with torch.no_grad():

        for entry_idx in trange(entry_count):
            if embed is not None:
                # generated embeddings are concatenated.
                # at start = (B, 10, 768)
                generated = embed  
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            # each iteration, the context on which generation is conditioned increases by 1. (prefix + gpt_outputs)
            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits

                # Using the output of time step and concatenating it with previous outputs. this (generated) is fed to gpt at next time step. 

                #we consider the logit of the last token
                # the logit of last token has the encoded context of all the previous tokens.
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)  # (B, T, vocab_size)
               
               # ----------------------------
                # """
                # TAKING THE LARGEST LOGIT
                # next_token = torch.argmax(logits, dim = -1).unsqueeze(0)

                # """
                # ----------------------------
               
                # """                
                # torch.multinomial

                probs = torch.nn.functional.softmax(logits, dim=-1) # (B, C)
                # sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
                
                # """
               # ---------------------------- 
                """
                THEIR IMPLEMENTATION
                # sort logits in descending order.
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                # take softmax across all the vocab_tokens --> They all sum to one and are arranged in descendng order
                # take cumulative sum. [0.4, 0.3,0.2,0.1] --> [0.4,0.7,0.9,1]
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                # The first few vocab tokens have the highest probability. For ex: out of 50k tokens, first 4 might comprise of .8 cumulative prob.
                # we retain only those tokens
                sorted_indices_to_remove = cumulative_probs > top_p
                # currently, don't drop first k indices
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                # #now, don't drop first k+1 elements
                sorted_indices_to_remove[..., 0] = 0  # isn't this redundant??
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                # The logits has a score for each vocab token. For indices_to_remove, that score = -inf
                # while taking softmax, those indices are ignored
                logits[:, indices_to_remove] = filter_value
                # Find idx (vocab token) for which the logit score is highest.
                
                next_token = torch.argmax(logits, -1).unsqueeze(0) # idx for a vocab token (B, 1)
                """
                # ----------------------------

                # for the given idx, go the nn.Embedding table and get token embedding for this idx
                next_token_embed = model.gpt.transformer.wte(next_token) # (B,1,768)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                #the generated embeddings are increased by 1 at each iteration. 
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

with torch.no_grad():


    model = Model(clip_dim = prefix_dim, prefix_len = prefix_length, const_len =prefix_length_clip, num_layers = num_layers,only_projection =True)

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model = model.eval()
    model = model.to(device)  

    # Get normalized clip embeddings for val set
    prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
    prefix = prefix / prefix.norm(2, -1).item()

    # Get prefix using mapping network
    prefix_embed = model(prefix).reshape(1, prefix_length, -1)
    
    if use_beam_search:
        generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
    else:
        generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)


print('\n')
print(generated_text_prefix)
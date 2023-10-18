import torch
from torch import nn, einsum

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F



class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support bias, simply bias=False """

    def __init__(self, dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else 0

    def forward(self, x):
        # return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        x_normalized = (x - mean) * (var + eps).rsqrt()
        return x_normalized * self.weight + self.bias
    

# https://jalammar.github.io/illustrated-transformer/
# nice website to read up on how transformers and attention works
class CausalSelfAttention(nn.Module):
    """ Multi-Head attention mechanism """

    def __init__(
        self, 
        emb_dim, 
        max_length, 
        num_heads=4, 
        head_dim=None, 
        dropout=0.1, 
        masked=True
    ):
        super().__init__()
        
        # set the number of heads we want for multi-head attention
        self.num_heads = num_heads

        # head_dim is the new dimensions we want for each word embedding so to achieve 
        # multi-head attention, multiply this by the desired number of heads
        hidden_dim = emb_dim
        if head_dim is not None:
            hidden_dim = head_dim * num_heads
            
        # create the matrices needed to compute the query, key, and value vectors
        self.c_attn = nn.Linear(emb_dim, hidden_dim * 3)

        # one last mlp to combine all information from all the heads
        self.c_proj = nn.Linear(hidden_dim, emb_dim)

        # dropouts for qkv computing and multi-head attention projection
        self.dropout = nn.Dropout(dropout)
        self.mha_dropout = nn.Dropout(dropout)

        # if masked attention, we need to make sure that at each timestep, we can't see information
        # from future timesteps
        self.masked = masked
        self.register_buffer(
            'bias', 
            torch.tril(torch.ones(max_length, max_length)).view((1, 1, max_length, max_length)),
        )


    def forward(self, x, attn_mask=None):
        # batch, position (timestep), dimension of word embedding
        b, t, c = x.shape

        # get the query, value, and key vectors
        qkv = self.c_attn(x).chunk(3, dim = 2)

        # rearrange the vectors so that we separate by each head
        # new shape = batch, head, position(timestep), head_dim
        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h = self.num_heads), qkv)

        # matrix multiply the query and key vectors to get a table of the dot products
        # of each query and key vector at every pair of timesteps in the given block
        # i is the dimension for the query vector, and k is the dimenstion for the key vector
        # scale this product by the square root of the dimension of the vectors
        scores = einsum('b h i d, b h j d -> b h i j', q, k) / (k.size(-1) ** 0.5)

        # if there is masking, we need to mask the scores from future timesteps
        if self.masked:
            # since the matrix is lower triangular, for each query, we can only see 
            # key vectors from the same or a previous timestep
            scores = scores.masked_fill(self.bias[:,:,:t,:t] == 0, torch.finfo(scores.dtype).min)

        if attn_mask is not None:
            # this attention mask adds an infinitely negative number to the padded positions
            # so they don't get any attention
            scores += attn_mask

        # softmax over the last dimension (the dimension of the key vectors)
        attn = scores.softmax(dim = -1)
        attn = self.dropout(attn)

        # basically a linear combination where attn is the weights and v is the values
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # concatenate all the heads now
        out = rearrange(out, 'b h t d -> b t (h d)')
        out = self.mha_dropout(self.c_proj(out))
        
        return out
    
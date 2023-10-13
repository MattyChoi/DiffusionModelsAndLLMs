import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import LayerNorm, CausalSelfAttention


class FFN(nn.Module):
    """ 
    Feed Forward layer a simple linear layer followed by a non-linearity 
    """

    def __init__(self, emb_dim, dropout=0.1):
        super().__init__()

        self.fc = nn.Linear(emb_dim, 4 * emb_dim)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(4 * emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x
    

class DecoderBlock(nn.Module):
    """ 
    GPT uses only a decoder block and no encoder
    """

    def __init__(self, emb_dim, max_length, num_heads=4):
        """
        dim: the current dimension of the word embeddings
        max_length: the maximum length of timesteps
        num_heads: number of attention heads
        """
        super().__init__()

        self.ln1 = LayerNorm(emb_dim)
        self.attn = CausalSelfAttention(
            emb_dim, 
            max_length, 
            num_heads=num_heads, 
            head_dim = 32, 
            dropout=0.1, 
            masked=True
        )
        self.ln2 = LayerNorm(emb_dim)
        self.mlp = FFN(emb_dim)

    def forward(self, x):
        # https://datascience.stackexchange.com/questions/85486/what-is-the-difference-between-gpt-blocks-and-transformer-decoder-blocks
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

class GPT(nn.Module):
    def __init__(
        self, 
        vocab_size, # 28996 for e2e_nlg, so use 32768
        max_length, 
        emb_dim,    # 
        num_heads, 
        num_layers,
    ):
        """
        vocab_size: number of tokens in the dataset
        max_length: the maximum length of timesteps
        emb_dim: number of dimensions you want for each word embedding
        num_heads: number of heads for self attention
        num_layers: number of decoder blocks
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding_table = nn.Embedding(max_length, emb_dim)
        self.blocks = nn.Sequential(*[DecoderBlock(emb_dim, n_head=num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(emb_dim) # final layer norm
        self.lm_head = nn.Linear(emb_dim, vocab_size)
        
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, index, targets=None):
        print(index.shape)
        B, T = index.shape
        
        
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(index) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=index.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            # get the predictions
            logits, loss = self.forward(index)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)

        return index
import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.text_gen.attention import LayerNorm, CausalSelfAttention


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

    def forward(self, x, attn_mask=None):
        # https://datascience.stackexchange.com/questions/85486/what-is-the-difference-between-gpt-blocks-and-transformer-decoder-blocks
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x
    

class GPT2LMHeadModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        max_length: int, 
        emb_dim: int,
        num_heads: int, 
        num_layers: int,
        dropout: float = 0.1,
    ):
        """
        vocab_size: number of tokens in the dataset
        max_length: the maximum length of timesteps
        emb_dim: number of dimensions you want for each word embedding
        num_heads: number of heads for self attention
        num_layers: number of decoder blocks
        """
        super().__init__()

        self.max_length = max_length

        self.transformer = nn.ModuleDict(dict(
            # create word embeddings from tokens
            token_emb = nn.Embedding(vocab_size, emb_dim),

            # create positional encdoings
            pos_emb = nn.Embedding(max_length, emb_dim),
            dropout = nn.Dropout(dropout),

            # decoder blocks
            decoder = nn.ModuleList(
                [
                    DecoderBlock(
                        emb_dim=emb_dim, 
                        max_length=max_length,
                        num_heads=num_heads) 
                    for _ in range(num_layers)
                ]
            ),

            # final layer norm
            ln = nn.LayerNorm(emb_dim)
        ))

        # text generation head
        self.head = nn.Linear(emb_dim, vocab_size)
        
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, attn_mask=None, labels=None):
        """
        idx: outputs of a string of words through a tokenizer, should have shape (batch_size, vocab_size)
        attn_mask: vector of the same shape as idx with 0s for pad tokens and 0s for the rest
        targets: same as idx, should have shape (batch_size, vocab_size)
        """
        # get the device the inputs are being trained on
        device = idx.device

        # b is the batch size, t is the number of tokens
        b, t = idx.shape

        assert t <= self.max_length, f"Cannot forward sequence of length {t}, block size is only {self.max_length}"

        pos = torch.arange(t, dtype=torch.long, device=device)

        # get the word and positional embeddings
        tok_emb = self.transformer.token_emb(idx) # shape is (b, t, emb_dim)
        pos_emb = self.transformer.pos_emb(pos) # shape is (t, c)

        # put through dropout
        x = self.transformer.dropout(tok_emb + pos_emb) # shape is (b, t, emb_dim)

        # create the attention mask for the causal attention mechanism
        if attn_mask is not None:
            attn_mask = attn_mask.view(b, -1)       # make sure it's the same shape as the tokens
            attn_mask = attn_mask[:, None, None, :]
            attn_mask = (1.0 - attn_mask) * torch.finfo(x.dtype).min

        # put it through the decoder blocks
        for block in self.transformer.decoder:
            x = block(x, attn_mask) # shape is (b, t, emb_dim)

        # apply the last layer norm
        x = self.transformer.ln(x) # shape is (b, t, emb_dim)

        loss = None
        if labels is not None:
            # get the scores for each vocab
            logits = self.head(x) # shape is (b, t, vocab_size)

            # shift the logits and labels so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # combine the batch and timestep axes for better parallelization
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1), 
            )
        else:
            # if there is no desired target, just use the logits from the last time step
            logits = self.head(x)[:, [-1], :] # shape is (b, 1, vocab_size)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, attn_mask=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        b, t = idx.shape
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_crop = idx if idx.size(1) <= self.max_length else idx[:, -self.max_length:]

            # get the logits for each batch
            logits, _ = self.forward(idx_crop, attn_mask)  # shape is (b, 1, vocab_size)

            # focus only on the last time step although this is already done for 
            # inference without any targets
            logits = logits[:, -1, :] # shape is (b, vocab_size)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (b, vocab_size)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (b, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

            # update attention mask
            if attn_mask is not None:
                ind = torch.sum(attn_mask, dim=1, dtype=torch.long)
                batch_inds = ind < t
                ind = ind[batch_inds]
                attn_mask[batch_inds.nonzero(), ind] = 1.0
                
        return idx
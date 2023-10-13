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

        # create word embeddings from tokens
        self.token_emb = nn.Embedding(vocab_size, emb_dim)

        # create positional encdoings
        self.pos_emb = nn.Embedding(max_length, emb_dim)

        self.dropout = nn.Dropout(dropout)

        # decoder blocks
        self.decoder = nn.Sequential(
            *[
                DecoderBlock(
                    emb_dim=emb_dim, 
                    max_length=max_length,
                    num_heads=num_heads) 
                for _ in range(num_layers)
            ]
        )

        # final layer norm
        self.ln = nn.LayerNorm(emb_dim)
        
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


    def forward(self, idx, targets=None):
        """
        idx: outputs of a string of words through a tokenizer, should have shape (batch_size, vocab_size)
        targets: same as idx, should have shape (batch_size, vocab_size)
        """
        # get the device the inputs are being trained on
        device = idx.device

        # b is the batch size, t is the number of tokens
        b, t = idx.shape

        assert t <= self.max_length, f"Cannot forward sequence of length {t}, block size is only {self.max_length}"

        pos = torch.arange(t, dtype=torch.long, device=device)

        # get the word and positional embeddings
        tok_emb = self.token_emb(idx) # shape is (b, t, emb_dim)
        pos_emb = self.pos_emb(pos) # shape is (t, c)

        # put through dropout
        x = self.dropout(tok_emb + pos_emb) # shape is (b, t, emb_dim)

        # put it through the decoder blocks
        x = self.decoder(x) # shape is (b, t, emb_dim)

        # apply the last layer norm
        x = self.ln(x) # shape is (b, t, emb_dim)

        
        loss = None
        if targets is not None:
            # get the scores for each vocab
            logits = self.head(x) # shape is (b, t, vocab_size)

            vocab_size = logits.size(1)

            # combine the batch and timestep axes for better parallelization
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-1)
        else:
            # if there is no desired target, just use the logits from the last time step
            logits = self.head(x)[:, [-1], :] # shape is (b, 1, vocab_size)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_crop = idx if idx.size(1) <= self.max_length else idx[:, -self.max_length:]

            # get the logits for each batch
            logits, _ = self.forward(idx_crop)  # shape is (b, 1, vocab_size)

            # focus only on the last time step although this is already done for 
            # inference without any targets
            logits = logits[:, -1, :] # shape is (b, vocab_size)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (b, vocab_size)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (b, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.text_gen.attention import CausalSelfAttention
from transformers import AutoTokenizer


class FFN(nn.Module):
    """ 
    Feed Forward layer a simple linear layer followed by a non-linearity 
    """

    def __init__(self, emb_dim, dropout=0.1):
        super().__init__()

        self.c_fc = nn.Linear(emb_dim, 4 * emb_dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class DecoderBlock(nn.Module):
    """ 
    GPT uses only a decoder block and no encoder
    """

    def __init__(self, emb_dim, max_length, num_heads=4, layer_norm_epsilon=1e-05):
        """
        dim: the current dimension of the word embeddings
        max_length: the maximum length of timesteps
        num_heads: number of attention heads
        """
        super().__init__()

        self.ln_1 = nn.LayerNorm(emb_dim, eps=layer_norm_epsilon)
        self.attn = CausalSelfAttention(
            emb_dim, 
            max_length, 
            num_heads=num_heads, 
            head_dim=None, 
            dropout=0.1, 
            masked=True
        )
        self.ln_2 = nn.LayerNorm(emb_dim)
        self.mlp = FFN(emb_dim)

    def forward(self, x, attn_mask=None):
        # https://datascience.stackexchange.com/questions/85486/what-is-the-difference-between-gpt-blocks-and-transformer-decoder-blocks
        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
    

class GPT2LM(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        max_length: int, 
        emb_dim: int,
        tokenizer,
        num_heads: int, 
        num_layers: int,
        dropout: float = 0.1,
    ):
        """
        vocab_size: number of tokens in the dataset
        max_length: the maximum length of timesteps
        emb_dim: number of dimensions you want for each word embedding
        tokenizer: to get the tokenizer eos and pad tokens
        num_heads: number of heads for self attention
        num_layers: number of decoder blocks
        """
        super().__init__()

        self.max_length = max_length
        self.tokenizer = tokenizer

        self.transformer = nn.ModuleDict(dict(
            # create word embeddings from tokens
            wte = nn.Embedding(vocab_size, emb_dim),

            # create positional encdoings
            wpe = nn.Embedding(max_length, emb_dim),
            dropout = nn.Dropout(dropout),

            # decoder blocks
            h = nn.ModuleList(
                [
                    DecoderBlock(
                        emb_dim=emb_dim, 
                        max_length=max_length,
                        num_heads=num_heads) 
                    for _ in range(num_layers)
                ]
            ),

            # final layer norm
            ln_f = nn.LayerNorm(emb_dim)
        ))

        # text generation head
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)
        
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def crop_block_size(self, new_max_length):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert new_max_length <= self.max_length
        self.max_length = new_max_length
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:new_max_length])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:new_max_length,:new_max_length]


    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # look for available pretrained models
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict

        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(num_layers=12, num_heads=12, emb_dim=768),  # 124M params
            'gpt2-medium':  dict(num_layers=24, num_heads=16, emb_dim=1024), # 350M params
            'gpt2-large':   dict(num_layers=36, num_heads=20, emb_dim=1280), # 774M params
            'gpt2-xl':      dict(num_layers=48, num_heads=25, emb_dim=1600), # 1558M params
        }[model_type]
        
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['max_length'] = 1024 # always 1024 for GPT model checkpoints
        config_args['tokenizer'] = AutoTokenizer.from_pretrained(
            "gpt2",
            use_fast=True,
            use_auth_token=False,
            pad_token="<|endoftext|>",
            padding_side="left"
        )

        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        # create a from-scratch initialized minGPT model
        model = cls(**config_args)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        from transformers import GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()

        # ignore the masked bias buffers in the attnetion module
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    

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
        tok_emb = self.transformer.wte(idx) # shape is (b, t, emb_dim)
        pos_emb = self.transformer.wpe(pos) # shape is (t, c)

        # put through dropout
        x = self.transformer.dropout(tok_emb + pos_emb) # shape is (b, t, emb_dim)

        # create the attention mask for the causal attention mechanism
        if attn_mask is not None:
            attn_mask = attn_mask.view(b, -1)       # make sure it's the same shape as the tokens
            attn_mask = attn_mask[:, None, None, :]
            attn_mask = (1.0 - attn_mask) * torch.finfo(x.dtype).min

        # put it through the decoder blocks
        for block in self.transformer.h:
            x = block(x, attn_mask) # shape is (b, t, emb_dim)

        # apply the last layer norm
        x = self.transformer.ln_f(x) # shape is (b, t, emb_dim)

        loss = None
        # get the scores for each vocab
        logits = self.lm_head(x) # shape is (b, t, vocab_size)
        if labels is not None:

            # shift the logits and labels so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # combine the batch and timestep axes for better parallelization
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1), 
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, attn_mask=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        b, t = input_ids.shape

        # use the eos_token to know when a sequence is done generating
        eos_token_id = self.tokenizer.eos_token_id

        # use the pad_token to pad the remainder of the sentence
        pad_token_id = self.tokenizer.pad_token_id

        # create a tensor for the eos_token
        eos_token_tensor = torch.tensor([eos_token_id]).to(input_ids.device)

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(b, dtype=torch.long, device=input_ids.device)

        for _ in range(max_new_tokens):
            # get the logits for each batch
            logits, _ = self.forward(input_ids, attn_mask)  # shape is (b, 1, vocab_size)

            # focus only on the last time step although this is already done for 
            # inference without any targets
            logits = logits[:, -1, :] # shape is (b, vocab_size)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (b, vocab_size)

            # find the most likely token
            next_tokens = torch.argmax(probs, dim=-1, keepdim=True) # (b, 1)
            # # or sample from the distribution
            # next_tokens = torch.multinomial(probs, num_samples=1) # (b, 1)
            
            # finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # append sampled index to the running sequence
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

            # update attention mask
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [
                        attn_mask, 
                        attn_mask.new_ones((b, 1))
                    ], 
                    dim=-1,
                )

            # crop the inputs and attn_mask
            if input_ids.size(-1) > self.max_length:
                input_ids = input_ids[:, -self.max_length:]
                attn_mask = attn_mask[:, -self.max_length:]

            # if eos_token was found in a sentence, set the sentence to being finished
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.ne(eos_token_tensor)
            )

            # break the loop if all sentences are finished
            if unfinished_sequences.max() == 0:
                break

                
        return input_ids
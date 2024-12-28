from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math 
import inspect 
import os
import time
import numpy as np
# ----------------------------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        #key, query and value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        #regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        #not really a bias, more of a mask, but following the OpenAI naming convention
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() #batch size, sequence length, embedding dimensionality (n_embd)
        #calculate query, key, value for all heads in batch and move head forward to be the batch dimension
        #nh is the number of heads, hs is the head size, and C (number of channels) = nh * hs
        #e.g. in GPT-2 (124M): nh = 12, hs = 64, C = 768 in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T, T) attention matrix) for all the queries and keys
        
        #we will use flash attention here
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) #query times key, scaled by the square root of key size. This gives us the raw attention similarity scores.
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) #auto-regressive mask to prevent attending to future tokens during training
        # att = F.softmax(att, dim=-1) #normalize the attention scores
        # y = att @ v # (B, nh, T, hs) x (B, nh, T, hs) -> (B, nh, T, hs)



        y = y.transpose(1, 2).contiguous().view(B, T, C) #re-assemble all head outputs side by side
        #output projection
        y = self.c_proj(y)
        return y




class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4) #Linear projection of the input to a higher dimensionality
        self.gelu = nn.GELU(approximate='tanh') #problem with ReLU is that it can kill gradients during the backward pass if the tail of the ReLU is zero. GELU is a smooth version of ReLU.
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd) #Linear projection of the output back to the original dimensionality
        self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        #We would prefer a clean residual stream because addition distributes the gradients during the backward pass to both of its branches equally. 


       

    def forward(self, x): 
        x = x + self.attn(self.ln_1(x)) #x goes through first the normalization layer, then the attention layer.
        x = x + self.mlp(self.ln_2(x)) #x goes through the second normalization layer, then the MLP and then to the residual stream.
        return x
    
@dataclass

#GPT-2 has only a decoder, no encoder
class GPTConfig:
    block_size: int = 1024 #maximum sequence length
    vocab_size: int = 50257 #number of tokens in the vocabulary
    n_layer: int = 12 #number of layers in the Transformer
    n_head: int = 12 #number of heads in the multi-head attention
    n_embd: int = 768 #dimension of the embeddings and the model

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # nn.Embedding creates a lookup table where:
            # - num_embeddings: size of the dictionary (number of unique tokens/words)
            # - embedding_dim: size of the vector that each token will be mapped to

            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #Stack of blocks
            ln_f = nn.LayerNorm(config.n_embd), #Layer Normalization 
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) #language model head - maps the output of the transformer to the vocabulary

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        #init params
        self.apply(self._init_weights) #iterates  all the submodules and initializes the weights
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 #the 2 times comes from the every layer in the transformer having two sub-layers: the attention and the feedforward 
            torch.nn.init.normal_(module.weight, mean = 0.0, std=std) #if the module is a linear layer, initialize the weights with a normal distribution
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) #if the module has a bias, initialize it with zeros
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std=0.02) #typically standard deviation is 1/sqrt(d) where d is the dimensionality of the embedding, but this is a good estimate.
        
    def forward(self,idx, targets = None):
        #idx is of shape (B, T) where B is the batch dimension and T is the time dimension
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T} block size is {self.config.block_size}"
        #forward the token and position embeddings
        pos = torch.arange(0,T, dtype=torch.long, device=idx.device) #shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd) -- Get the position embeddings
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd) -- Convert input tokens to embeddings
        x = tok_emb + pos_emb #Combine the token embeddings and the position embeddings -- shape (B, T, n_embd)
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x) #Each block includes attention and feedforward
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x) # Final layer normalization
        logits = self.lm_head(x) # Shape: (B, T, vocab_size) -- logits are a softmax away from probabilities
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) #flattening out 3D logits and 3D targets to 2D and calculating the loss
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        #n_layer, n_head and n_embd are determined by the model type'
        config_args = {
            'gpt2':    dict(n_layer=12, n_head=12, n_embd=768), #124M
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), #355M
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), #762M
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600), #1558M
        }[model_type]
        config_args['vocab_size'] = 50257 #always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 #always 1024 for GPT model checkpoints
        #create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] #discard this mask 

        # init a huggingfact/transfomers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        #copy while ensuring all of the parameters are aligned and match in names and shapes 
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
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
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        #start with all of the candidate parameters (that require gradients)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        #create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        #i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms do not
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim()< 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters") #2D tensors are weights
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters") #1D tensors are biases and layernorm weights
        #Create AdamW optimizer and use the fused version if possible
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters #checks if the fused parameter is available in the AdamW optimizer
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
        
# ----------------------------------------------------------------------------------------------------
import tiktoken 

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self,B,T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")

        #state
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = buf[:-1].view(B, T) #inputs
        y = buf[1:].view(B, T) #targets
        #advance the position in the tensor
        self.current_position += B*T * self.num_processes
        # if loading the next batch is out of bounds, reset the position
        if (self.current_position + B*T*self.num_processes + 1) >= len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y


    #     #get shard filenames
    #     data_root = "edu_fineweb10B"
    #     shards = os.listdir(data_root)
    #     shards = [s for s in shards if split in s] #filter out the shards that don't match the split
    #     shards = sorted(shards)
    #     shards = [os.path.join(data_root, s) for s in shards] #prepend the root path to the shard filenames
    #     self.shards = shards
    #     assert len(shards) > 0, f"no shards found for split {split}"
    #     if master_process:
    #         print(f"found {len(shards)} shards for split {split}") #print the number of shards found 
    #     self.reset()

    # def reset(self):    
    #     #state, init at shard zero
    #     self.current_shard = 0
    #     self.tokens = load_tokens(self.shards[self.current_shard])
    #     self.current_position = self.B * self.T * self.process_rank #start at the beginning of the shard

    # def next_batch(self):
    #     B, T = self.B, self.T
    #     buf = self.tokens[self.current_position : self.current_position + B*T + 1]
    #     x = (buf[:-1]).view(B, T) #inputs
    #     y = (buf[1:]).view(B, T) #targets
    #     #advance the position in the tensor 
    #     self.current_position += B * T * self.num_processes
    #     # if loading the next batch is out of bounds, reset the position
    #     if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
    #         self.current_shard = (self.current_shard + 1) % len(self.shards)
    #         self.tokens = load_tokens(self.shards[self.current_shard])
    #         self.current_position = B * T * self.process_rank
    #     return x, y

# ----------------------------------------------------------------------------------------------------
#DDP launch for e.g. 4 GPUs:
#torchrun --standalone --nproc_per_node=4 gpt2_cuda.py

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "for now I think we need CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    #device = torch.device('cuda', ddp_local_rank)
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

torch.manual_seed(1337) #for reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288 #2**19, ~0.5M, in number of tokens
B = 16 #microbatch size
T = 1024 #sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure the total batch size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) #number of steps to accumulate gradients over
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# print("I am GPU", ddp_rank)
# print("Bye")
# import sys; sys.exit(0)

train_loader = DataLoaderLite(B=B, T=T, process_rank = ddp_rank, num_processes = ddp_world_size, split='train')
val_loader = DataLoaderLite(B=B, T=T, process_rank = ddp_rank, num_processes = ddp_world_size, split='val')
torch.set_float32_matmul_precision('high') 

#create model
model = GPT(GPTConfig(vocab_size = 50304))
model.to(device)
use_compile = False #torch.compile interferees with HellaSwag eval and Generation.
if use_compile:
    model = torch.compile(model) #broken on Apple Silicon M-series(?)

if ddp:
    model = DDP(model,device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model #always contains the "raw" unwrapped model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 #GPT3 paper says warmup of 375M tokens, so 375e6 / 524288 (2**19)
max_steps = 19073 #10e9 tokens / 524288 tokens per batch
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) #coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

#optimize!
#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8) #AdamW is a bugfix of Adam
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

#create log dir to write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: #open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    #once in a while evaluate validation loss
    if step % 25 == 0 or last_step:
        model.eval()
        #val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")    

    #once in a while generate from the model (except step 0, which is just random noise)
    #disabled because torch.compile throws an error 
    #if you disable torch.compile, code works fine
    


    if ((step > 0 and step % 25 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,") 
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank) #different seed for each process
        while xgen.size(1) < max_length:
            #forward the model to get logits 
            with torch.no_grad():
                logits, loss = model(xgen) # (B, T, vocab_size)
                #take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1) # (B, vocab_size)
                #do top-k sampling of 50 (huggingface pipeline default)
                #topk_probs here becomes (5, 50), topk_indices becomes (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) #anything lower than 50 is clamped to 0 
                #select a token from the top-k probabilities
                ix = torch.multinomial(topk_probs, 1, generator = sample_rng) # (B, 1)
                #gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                #append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1) # (B, T)
        #print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank}: sample {i}: {decoded}")

    #do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #gradient clipping
    #determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() #waits for all the operations to finish
    t1 = time.time()
    dt = (t1 - t0) 
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt #throughput
    if master_process:
        print(f"step {step:4d} |  loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f} ms, tokens/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
if ddp:
    destroy_process_group()













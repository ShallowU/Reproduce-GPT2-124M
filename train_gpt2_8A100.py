import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import numpy as np
from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# CausalSelfAttention, MLP, Block, GPTConfig, GPT, DataLoaderLite, get_most_likely_row
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass # this is a dataclass, it will automatically generate __init__, __repr__, __eq__ etc.
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):  # 保持梯度在合理范围内，避免梯度消失和爆炸
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
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

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# 添加数据下载函数
def download_dataset():
    """下载数据集到本地"""
    if not os.path.exists("edu_fineweb10B"):
        print("Downloading dataset from Hugging Face...")
        download_start_time = time.time()
        snapshot_download(
            repo_id="ShallowU/FineWeb-Edu-10B-Tokens-NPY",
            repo_type="dataset", 
            local_dir="edu_fineweb10B"
        )
        download_end_time = time.time()
        download_duration = download_end_time - download_start_time
        print("Dataset downloaded successfully!")
        return download_duration
    else:
        print("Dataset already exists locally.")
        return 0.0

# 添加生成测试函数
def generate_final_samples(model, enc, device, device_type, output_file):
    """训练结束后生成样本"""
    model.eval()
    prompts = [
        "Hello, I'm a language model,",
        "Hello, I'm a computer science student,"
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== Final Model Generation Results ===\n\n")
        
        for prompt in prompts:
            f.write(f"Prompt: {prompt}\n")
            f.write("-" * 50 + "\n")
            
            for i in range(3):  # 每个提示词生成3次
                tokens = enc.encode(prompt)
                tokens = torch.tensor(tokens, dtype=torch.long)
                tokens = tokens.unsqueeze(0).to(device)  # (1, T)
                
                sample_rng = torch.Generator(device=device)
                sample_rng.manual_seed(42 + i)  # 不同的随机种子
                
                max_length = 64  # 更长的生成长度
                while tokens.size(1) < max_length:
                    with torch.no_grad():
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            logits, _ = model(tokens)
                        logits = logits[:, -1, :]
                        probs = F.softmax(logits, dim=-1)
                        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                        ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                        xcol = torch.gather(topk_indices, -1, ix)
                        tokens = torch.cat((tokens, xcol), dim=1)
                
                decoded = enc.decode(tokens[0].tolist())
                f.write(f"Generation {i+1}: {decoded}\n\n")
            
            f.write("\n" + "="*70 + "\n\n")

# 添加绘制损失函数
def plot_losses(log_file, output_dir):
    """从日志文件绘制训练和验证损失"""
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []
    
    with open(log_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                step = int(parts[0])
                loss_type = parts[1]
                loss_value = float(parts[2])
                
                if loss_type == 'train':
                    train_steps.append(step)
                    train_losses.append(loss_value)
                elif loss_type == 'val':
                    val_steps.append(step)
                    val_losses.append(loss_value)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_steps, train_losses, 'b-', label='Training Loss', alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_steps, val_losses, 'r-', label='Validation Loss', alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 也绘制一个合并的图
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_losses, 'b-', label='Training Loss', alpha=0.7)
    plt.plot(val_steps, val_losses, 'r-', label='Validation Loss', alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'combined_loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 主训练代码
if __name__ == "__main__":
    total_start_time = time.time()
    # 下载数据集
    download_time=download_dataset()
    
    # DDP 设置
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    enc = tiktoken.get_encoding("gpt2")

    # 训练参数
    total_batch_size = 524288
    B = 32  # 32 for 2 A100(40G) and  use_compile =true
    T = 1024
    assert total_batch_size % (B * T * ddp_world_size) == 0
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # 数据加载器
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

    torch.set_float32_matmul_precision('high')

    # 创建模型
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    
    # 启用编译以加速训练
    use_compile = False # 自己可以先实验是否会加速训练再决定是否开启
    if use_compile:
        model = torch.compile(model)
    
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # 学习率调度
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    max_steps = 19073

    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    # 优化器
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

    # 日志目录
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f:
        pass

    training_start_time = time.time()
    if master_process:
        print(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_start_time))}")
        print(f"Total steps: {max_steps}")
    # 训练循环
    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # 验证损失评估
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 5000 == 0 or last_step):
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item()
                    }
                    torch.save(checkpoint, checkpoint_path)

        # 训练步骤
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    # 记录训练结束时间
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    total_duration = training_end_time - total_start_time
    # 训练结束后的处理
    if ddp:
        destroy_process_group()
    
    if master_process:
        print("Training completed! Processing results...")
        
        # 绘制损失曲线
        plot_losses(log_file, log_dir)
        print("Loss curves saved!")
        
        # 生成最终样本
        generation_file = os.path.join(log_dir, "final_generations.txt")
        generate_final_samples(raw_model, enc, device, device_type, generation_file)
        print("Final generations saved!")
    
        print(f"Training finished at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_end_time))}")
        print("\n📊 Timing Summary:")
        print(f"  Data Download Time: {download_time:.2f} seconds ({download_time/60:.2f} minutes)")
        print(f"  Pure Training Time: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")
        print(f"  Total Runtime: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        print(f"  Average time per step: {training_duration/max_steps:.3f} seconds")
        print(f"  Total tokens processed: {max_steps * total_batch_size:,}")
        print(f"  Average throughput: {(max_steps * total_batch_size) / training_duration:.0f} tokens/sec")
        
        # 保存时间统计到单独的文件
        timing_file = os.path.join(log_dir, "timing_summary.txt")
        with open(timing_file, "w") as f:
            f.write("GPT-2 Training Timing Summary\n")
            f.write("="*40 + "\n\n")
            f.write(f"Training started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_start_time))}\n")
            f.write(f"Training finished: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_end_time))}\n\n")
            f.write(f"Data Download Time: {download_time:.2f} seconds ({download_time/60:.2f} minutes)\n")
            f.write(f"Pure Training Time: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)\n")
            f.write(f"Total Runtime: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)\n\n")
            f.write(f"Training Configuration:\n")
            f.write(f"  Total steps: {max_steps:,}\n")
            f.write(f"  Batch size: {total_batch_size:,} tokens\n")
            f.write(f"  GPU count: {ddp_world_size}\n")
            f.write(f"  Model parameters: ~124M\n\n")
            f.write(f"Performance Metrics:\n")
            f.write(f"  Average time per step: {training_duration/max_steps:.3f} seconds\n")
            f.write(f"  Total tokens processed: {max_steps * total_batch_size:,}\n")
            f.write(f"  Average throughput: {(max_steps * total_batch_size) / training_duration:.0f} tokens/sec\n")
            f.write(f"  GPU utilization: {(max_steps * total_batch_size) / training_duration / ddp_world_size:.0f} tokens/sec/GPU\n")
        
        print("⏱️  Timing summary saved!")
        print("="*60)
        print("All results saved to log directory.")
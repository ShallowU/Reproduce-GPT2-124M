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

# 添加时间记录类
class TimeTracker:
    def __init__(self):
        self.times = {}
        self.start_times = {}
    
    def start_timing(self, name):
        """开始计时"""
        self.start_times[name] = time.time()
    
    def end_timing(self, name):
        """结束计时"""
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.times[name] = elapsed
            del self.start_times[name]
            return elapsed
        return 0
    
    def get_time_str(self, seconds):
        """将秒数转换为易读的时间格式"""
        if seconds < 60:
            return f"{seconds:.2f}秒"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{int(minutes)}分{secs:.1f}秒"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return f"{int(hours)}时{int(minutes)}分{secs:.1f}秒"
    
    def print_summary(self):
        """打印时间总结"""
        print("\n" + "="*60)
        print("⏱️  TIME SUMMARY REPORT")
        print("="*60)
        
        total_time = sum(self.times.values())
        
        for name, elapsed in self.times.items():
            percentage = (elapsed / total_time) * 100 if total_time > 0 else 0
            print(f"📊 {name:<25}: {self.get_time_str(elapsed):>15} ({percentage:>5.1f}%)")
        
        print("-" * 60)
        print(f"🕐 Total Execution Time    : {self.get_time_str(total_time):>15} (100.0%)")
        print("="*60)

# 全局时间跟踪器
timer = TimeTracker()

# [复制所有类定义：CausalSelfAttention, MLP, Block, GPTConfig, GPT, DataLoaderLite, get_most_likely_row]
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

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

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
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

def download_dataset():
    """下载数据集到本地"""
    timer.start_timing("Dataset Download")
    
    if not os.path.exists("edu_fineweb10B"):
        print("Downloading dataset from Hugging Face...")
        snapshot_download(
            repo_id="ShallowU/FineWeb-Edu-10B-Tokens-NPY",
            repo_type="dataset", 
            local_dir="edu_fineweb10B"
        )
        print("Dataset downloaded successfully!")
    else:
        print("Dataset already exists locally.")
    
    download_time = timer.end_timing("Dataset Download")
    print(f"📥 Dataset operation completed in: {timer.get_time_str(download_time)}")

def generate_final_samples(model, enc, device, device_type, output_file):
    """训练结束后生成样本"""
    timer.start_timing("Sample Generation")
    print("Generating final samples...")
    
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
            
            for i in range(3):
                tokens = enc.encode(prompt)
                tokens = torch.tensor(tokens, dtype=torch.long)
                tokens = tokens.unsqueeze(0).to(device)
                
                sample_rng = torch.Generator(device=device)
                sample_rng.manual_seed(42 + i)
                
                max_length = 64
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
    
    generation_time = timer.end_timing("Sample Generation")
    print(f"🔤 Sample generation completed in: {timer.get_time_str(generation_time)}")

def plot_losses(log_file, output_dir):
    """从日志文件绘制训练和验证损失"""
    timer.start_timing("Plot Generation")
    print("Plotting loss curves...")
    
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
    
    plot_time = timer.end_timing("Plot Generation")
    print(f"📈 Loss curves generated in: {timer.get_time_str(plot_time)}")

def test_model_functionality():
    """测试模型基本功能"""
    timer.start_timing("Model Testing")
    print("Testing model functionality...")
    
    # 测试模型创建
    config = GPTConfig(vocab_size=50304)
    model = GPT(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 测试前向传播
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # 测试输入
    batch_size = 4
    seq_len = 128
    x = torch.randint(0, 50304, (batch_size, seq_len), device=device)
    y = torch.randint(0, 50304, (batch_size, seq_len), device=device)
    
    with torch.no_grad():
        logits, loss = model(x, y)
    
    print(f"Forward pass successful. Logits shape: {logits.shape}, Loss: {loss.item():.4f}")
    
    test_time = timer.end_timing("Model Testing")
    print(f"🧪 Model functionality test completed in: {timer.get_time_str(test_time)}")
    return True

# 主训练代码
if __name__ == "__main__":
    # 开始总体计时
    timer.start_timing("Total Execution")
    
    print("="*50)
    print("STARTING GPT-2 TRAINING FLOW TEST")
    print("="*50)
    
    # 测试模型基本功能
    test_model_functionality()
    
    # 下载数据集
    print("\n" + "="*30)
    print("STEP 1: DOWNLOADING DATASET")
    print("="*30)
    download_dataset()
    
    print("\n" + "="*30)
    print("STEP 2: SETTING UP TRAINING")
    print("="*30)
    
    timer.start_timing("Training Setup")
    
    # 单GPU设置（不使用DDP）
    ddp = False
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    enc = tiktoken.get_encoding("gpt2")

    # 训练参数 - 测试配置
    total_batch_size = 32768  # 减少batch size以适应单GPU
    B = 16  # 减少micro batch size
    T = 512  # 减少序列长度
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # 数据加载器
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

    torch.set_float32_matmul_precision('high')

    print("\n" + "="*30)
    print("STEP 3: CREATING MODEL")
    print("="*30)
    
    # 创建模型
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    
    # 保存原始模型引用
    raw_model = model
    
    # 启用编译
    use_compile = True
    if use_compile:
        print("Compiling model...")
        compile_start = time.time()
        model = torch.compile(model)
        compile_time = time.time() - compile_start
        print(f"Model compiled successfully in: {timer.get_time_str(compile_time)}")

    # 学习率调度 - 测试参数
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 5  # 减少到5步
    max_steps = 50    # 总共50步测试

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

    setup_time = timer.end_timing("Training Setup")
    print(f"⚙️ Training setup completed in: {timer.get_time_str(setup_time)}")

    print("\n" + "="*30)
    print("STEP 4: STARTING TRAINING LOOP")
    print("="*30)
    
    # 开始训练计时
    timer.start_timing("Training Loop")
    validation_total_time = 0
    checkpoint_total_time = 0
    
    # 训练循环
    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # 更频繁的验证：每10步或最后一步
        if step % 10 == 0 or last_step:
            val_start = time.time()
            print(f"\n--- Validation at step {step} ---")
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 3  # 减少验证步数
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            
            val_time = time.time() - val_start
            validation_total_time += val_time
            print(f"validation loss: {val_loss_accum.item():.4f} (time: {val_time:.2f}s)")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            
            # 更频繁保存checkpoint：每20步或最后一步
            if step > 0 and (step % 20 == 0 or last_step):
                checkpoint_start = time.time()
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                torch.save(checkpoint, checkpoint_path)
                checkpoint_time = time.time() - checkpoint_start
                checkpoint_total_time += checkpoint_time
                print(f"Checkpoint saved: {checkpoint_path} (time: {checkpoint_time:.2f}s)")

        # 训练步骤
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        
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
        
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

    training_time = timer.end_timing("Training Loop")
    print(f"\n🚀 Training loop completed in: {timer.get_time_str(training_time)}")
    print(f"   - Validation time: {timer.get_time_str(validation_total_time)}")
    print(f"   - Checkpoint time: {timer.get_time_str(checkpoint_total_time)}")
    
    # 记录验证和checkpoint时间
    timer.times["Validation Total"] = validation_total_time
    timer.times["Checkpoint Saving"] = checkpoint_total_time

    print("\n" + "="*30)
    print("STEP 5: POST-PROCESSING")
    print("="*30)
    
    # 训练结束后的处理
    print("Training completed! Processing results...")
    
    # 绘制损失曲线
    plot_losses(log_file, log_dir)
    
    # 生成最终样本
    generation_file = os.path.join(log_dir, "final_generations.txt")
    generate_final_samples(raw_model, enc, device, device_type, generation_file)
    
    # 结束总体计时
    timer.end_timing("Total Execution")
    
    print("\n" + "="*50)
    print("TRAINING FLOW TEST COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("Files generated:")
    print(f"- Training log: {log_file}")
    print(f"- Loss curves: {os.path.join(log_dir, 'loss_curves.png')}")
    print(f"- Combined loss curves: {os.path.join(log_dir, 'combined_loss_curves.png')}")
    print(f"- Final generations: {generation_file}")
    print(f"- Checkpoints: {log_dir}/model_*.pt")
    
    # 打印时间总结
    timer.print_summary()
    
    print("\nYou can now test the upload script with: python3 upload_hf.py")
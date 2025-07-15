#!/usr/bin/env python3
# 生成GPT-2 124M模型的文本
# 该脚本下载模型并生成文本，支持交互式测试和自动化
# 需要安装tiktoken和huggingface_hub库
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from huggingface_hub import hf_hub_download, login
from dataclasses import dataclass

# 复制必要的模型类定义
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
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
    vocab_size: int = 50304  # 修正：与训练时一致
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
        # # start with all of the candidate parameters (that require grad)
        # param_dict = {pn: p for pn, p in self.named_parameters()}
        # param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        # decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        # nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        # optim_groups = [
        #     {'params': decay_params, 'weight_decay': weight_decay},
        #     {'params': nodecay_params, 'weight_decay': 0.0}
        # ]
        # num_decay_params = sum(p.numel() for p in decay_params)
        # num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # if master_process:
        #     print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        #     print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # # Create AdamW optimizer and use the fused version if it is available
        # fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # use_fused = fused_available and device_type == "cuda"
        # if master_process:
        #     print(f"using fused AdamW: {use_fused}")
        # optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        # return optimizer
        pass

def download_and_load_model():
    """下载并加载模型"""
    access_token = " "# your token to access
    login(token=access_token)
    
    print("🔄 Downloading model from HuggingFace...")
    
    try:
        model_path = hf_hub_download(
            repo_id="ShallowU/GPT2-124M",
            filename="model_10000.pt",
            repo_type="model"
        )
        print(f"✅ Model downloaded to: {model_path}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None
    
    print("🔄 Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 使用checkpoint中的config，如果没有则使用默认
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = GPTConfig()
    
    model = GPT(config)
    
    # 处理torch.compile的键名问题
    state_dict = checkpoint['model']
    
    # 检查并移除_orig_mod前缀
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        print("🔧 Removing torch.compile prefixes...")
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[10:]  # 移除'_orig_mod.'前缀
            else:
                new_key = key
            cleaned_state_dict[new_key] = value
        state_dict = cleaned_state_dict
    
    # 加载状态字典
    try:
        model.load_state_dict(state_dict)
        print("✅ Model weights loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return None
    
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Step: {checkpoint['step']}")
    print(f"   Validation Loss: {checkpoint['val_loss']:.4f}")
    print(f"   Config: {config}")
    print(f"   Device: {device}")
    
    return model, device

def generate_text(model, device, prompt, max_length=100, temperature=0.8, top_k=50):
    """生成文本"""
    enc = tiktoken.get_encoding("gpt2")
    
    # 编码提示词
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    print(f"🔄 Generating text for prompt: '{prompt}'")
    print(f"   Initial tokens: {len(tokens[0])}")
    endoftext_token_id = 50256
    # 生成循环
    with torch.no_grad():
        for _ in range(max_length):
            # 前向传播
            logits, _ = model(tokens)
            logits = logits[:, -1, :] / temperature  # 取最后一个token的logits并应用温度
            
            # Top-k采样
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                probs = F.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, 1)
                next_token = torch.gather(top_k_indices, -1, next_token_idx)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            
            # 添加新token
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # 检查是否生成了结束符或达到最大长度
            if next_token.item() == 50256: 
                break
                
    # 解码生成的文本
    generated_text = enc.decode(tokens[0].tolist())
    return generated_text

def test_model():
    """测试模型生成效果"""
    print("🚀 GPT-2 124M Model Testing")
    print("=" * 50)
    
    # 下载并加载模型
    result = download_and_load_model()
    if result is None:
        return
    
    model, device = result
    
    # 测试提示词
    prompts = [
        "Hello, I'm a language model,",
        "Hello, I'm a computer science student,",
        "The future of artificial intelligence",
        "In machine learning, we often",
        "Education is important because"
    ]
    
    print("\n🎯 Generation Results:")
    print("=" * 50)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n📝 Test {i}/5")
        print("-" * 30)
        
        # 生成3个不同的样本
        for j in range(3):
            print(f"\nSample {j+1}:")
            generated = generate_text(
                model=model,
                device=device, 
                prompt=prompt,
                max_length=50,  # 生成50个token
                temperature=0.8,
                top_k=50
            )
            
            # 只显示生成的部分（去掉原始提示词）
            generated_part = generated[len(prompt):].strip()
            print(f"💭 {prompt}{generated_part}")
        
        print("\n" + "="*50)
    
    print("\n✅ Testing completed!")

def interactive_test():
    """交互式测试"""
    print("\n🎮 Interactive Mode")
    print("Type your prompt and press Enter. Type 'quit' to exit.")
    
    result = download_and_load_model()
    if result is None:
        return
    
    model, device = result
    
    while True:
        prompt = input("\n📝 Enter your prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        if not prompt:
            continue
            
        try:
            generated = generate_text(
                model=model,
                device=device,
                prompt=prompt,
                max_length=80,
                temperature=0.8,
                top_k=50
            )
            
            print(f"\n🤖 Generated: {generated}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🔍 Choose testing mode:")
    print("1. Automated test with predefined prompts")
    print("2. Interactive test (you provide prompts)")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        test_model()
    elif choice == "2":
        interactive_test()
    else:
        print("❌ Invalid choice. Running automated test...")
        test_model()
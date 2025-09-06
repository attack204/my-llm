# 架构与文件说明

## k_model.py

ModelConfig(PretrainedConfig)
用途: 模型配置类，定义结构超参数（dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len, dropout 等），兼容 transformers 的配置体系。
依赖: transformers.PretrainedConfig。

RMSNorm(nn.Module)
用途: 实现 RMSNorm 归一化（无偏置、带可学习缩放 weight）。
依赖: torch, torch.nn.

Attention(nn.Module)
用途: 多头自注意力模块，支持分组 KV 头（n_kv_heads），支持 RoPE 旋转位置编码和可选的 Flash Attention（若 torch>=2 提供 scaled_dot_product_attention）。
关键步骤: 线性映射得到 Q,K,V → 应用 RoPE（apply_rotary_emb）→ 扩展 KV 头（repeat_kv）→ 自注意力（Flash 或手写带 causal mask）→ 输出投影与 dropout。
依赖: ModelConfig（超参）、apply_rotary_emb, repeat_kv, torch, torch.nn, torch.nn.functional as F.

MLP(nn.Module)
用途: 前馈网络（SwiGLU 变体：silu(w1(x)) * w3(x) 后接 w2），隐藏维自动按 multiple_of 对齐。
依赖: torch, torch.nn, torch.nn.functional.silu.

DecoderLayer(nn.Module)
用途: 单层解码器结构：RMSNorm → Attention 残差 → RMSNorm → MLP 残差。
依赖: Attention, MLP, RMSNorm, ModelConfig.

Transformer(PreTrainedModel)
用途: 完整的自回归解码器语言模型。包含嵌入、n_layers 个 DecoderLayer、最终 RMSNorm 和输出头；词嵌入与输出权重共享；提供 forward 计算 logits/损失，提供无 KV cache 的简单 generate。
依赖:
内部：DecoderLayer, RMSNorm, ModelConfig, precompute_freqs_cis（注册 freqs_cos/sin 缓存），torch, F.cross_entropy。
外部输出容器：transformers.CausalLMOutputWithPast。
基类：transformers.PreTrainedModel。

# Train


# Prepare ENV

- YUM

```bash
yum install tmux
```

```bash
# 1. 卸载残留
sudo nvidia-uninstall   # 如果上一步已部分安装

# 2. 下载并安装新版驱动
wget -c https://cn.download.nvidia.com/tesla/550.54.15/NVIDIA-Linux-x86_64-550.54.15.run
chmod +x NVIDIA-Linux-x86_64-550.54.15.run
sudo ./NVIDIA-Linux-x86_64-550.54.15.run \
  --disable-nouveau --no-questions --accept-license

```

- Python310 

```bash
sudo dnf groupinstall "Development Tools" -y
sudo dnf install -y gcc openssl-devel bzip2-devel libffi-devel zlib-devel 

cd /usr/src
sudo wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz
sudo tar xzf Python-3.10.12.tgz
cd Python-3.10.12

sudo ./configure --enable-optimizations
sudo make -j$(nproc)
sudo make altinstall


mkdir -p /root/.local/bin

ln -s /usr/local/bin/python3.10 ~/.local/bin/python3
ln -s /usr/local/bin/pip3.10 ~/.local/bin/pip3

echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```


```bash
python3 -m pip install --upgrade pip
python3 -m pip install --user transformers datasets tokenizers huggingface_hub

python3 -m pip install -r requirements.txt
```

# Pretrain

数据集大小: 32GB, 1300 万条原始样本

参数量 0.215B: 

| 符号          | 含义                   | 脚本默认值 |
| ------------- | ---------------------- | ---------- |
| d             | hidden dimension       | 1024       |
| V             | vocab size             | 49152      |
| L             | # transformer layers   | 18         |
| n_heads       | # attention heads      | 16         |
| max_seq_len   | 最大序列长度           | 2048       |
| d_ff          | MLP 中间维度           | 4096 (4×d) |



| 模块                       | 公式                | 数值（个）                |
| -------------------------- | ------------------- | ------------------------- |
| Token embedding            | V × d               | 49 152 × 1 024 = 50 331 648 |
| Positional embedding       | max_seq_len × d     | 2 048 × 1 024 = 2 097 152   |
| **LayerNorm（scale+shift）** | 每层 2d             | 18 × 2 × 1 024 = 36 864     |
| **Attention**              |                     |                           |
| ‑ QKV 投影                   | 3 × d × d           | 18 × 3 × 1 024² = 56 623 104 |
| ‑ 输出投影                   | d × d               | 18 × 1 024² = 18 874 368   |
| **Feed-forward**           |                     |                           |
| ‑ 第一层                     | d × d_ff            | 18 × 1 024 × 4 096 = 75 497 472 |
| ‑ 第二层                     | d_ff × d            | 18 × 4 096 × 1 024 = 75 497 472 |
| **Final LayerNorm**        | 2d                  | 2 × 1 024 = 2 048         |


10w token测试

batch32 10w => 40min
batch64 cuda out of memory
python3 ddp_pretrain.py --batch_size 32

LLM总参数量：215.127 百万
Epoch:[1/1](0/3125) loss:8.901 lr:0.0002000 epoch_Time:63.0min;
Epoch:[1/1](100/3125) loss:7.319 lr:0.0001995 epoch_Time:40.0min;
Epoch:[1/1](200/3125) loss:6.993 lr:0.0001982 epoch_Time:39.0min;
Epoch:[1/1](300/3125) loss:6.898 lr:0.0001959 epoch_Time:38.0min;
Epoch:[1/1](400/3125) loss:6.772 lr:0.0001928 epoch_Time:37.0min;
Epoch:[1/1](500/3125) loss:6.571 lr:0.0001889 epoch_Time:36.0min;
Epoch:[1/1](600/3125) loss:6.176 lr:0.0001841 epoch_Time:34.0min;
Epoch:[1/1](700/3125) loss:5.891 lr:0.0001786 epoch_Time:33.0min;
Epoch:[1/1](800/3125) loss:5.472 lr:0.0001724 epoch_Time:32.0min;
Epoch:[1/1](900/3125) loss:5.198 lr:0.0001656 epoch_Time:30.0min;
Epoch:[1/1](1000/3125) loss:5.015 lr:0.0001582 epoch_Time:29.0min;
Epoch:[1/1](1100/3125) loss:4.822 lr:0.0001504 epoch_Time:28.0min;
Epoch:[1/1](1200/3125) loss:4.816 lr:0.0001421 epoch_Time:26.0min;
Epoch:[1/1](1300/3125) loss:4.693 lr:0.0001335 epoch_Time:25.0min;
Epoch:[1/1](1400/3125) loss:4.577 lr:0.0001246 epoch_Time:23.0min;
Epoch:[1/1](1500/3125) loss:4.436 lr:0.0001157 epoch_Time:22.0min;
Epoch:[1/1](1600/3125) loss:4.240 lr:0.0001066 epoch_Time:21.0min;
Epoch:[1/1](1700/3125) loss:4.416 lr:0.0000976 epoch_Time:19.0min;
Epoch:[1/1](1800/3125) loss:4.190 lr:0.0000887 epoch_Time:18.0min;
Epoch:[1/1](1900/3125) loss:4.089 lr:0.0000800 epoch_Time:17.0min;
Epoch:[1/1](2000/3125) loss:4.219 lr:0.0000717 epoch_Time:15.0min;
Epoch:[1/1](2100/3125) loss:4.244 lr:0.0000637 epoch_Time:14.0min;
Epoch:[1/1](2200/3125) loss:3.995 lr:0.0000562 epoch_Time:12.0min;
Epoch:[1/1](2300/3125) loss:4.096 lr:0.0000492 epoch_Time:11.0min;
Epoch:[1/1](2400/3125) loss:3.938 lr:0.0000429 epoch_Time:10.0min;
Epoch:[1/1](2500/3125) loss:4.129 lr:0.0000372 epoch_Time:8.0min;
Epoch:[1/1](2600/3125) loss:3.983 lr:0.0000322 epoch_Time:7.0min;
Epoch:[1/1](2700/3125) loss:3.975 lr:0.0000281 epoch_Time:5.0min;
Epoch:[1/1](2800/3125) loss:3.883 lr:0.0000248 epoch_Time:4.0min;
Epoch:[1/1](2900/3125) loss:4.063 lr:0.0000223 epoch_Time:3.0min;
Epoch:[1/1](3000/3125) loss:4.087 lr:0.0000207 epoch_Time:1.0min;
Epoch:[1/1](3100/3125) loss:3.819 lr:0.0000200 epoch_Time:0.0min;

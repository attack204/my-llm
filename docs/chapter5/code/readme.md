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
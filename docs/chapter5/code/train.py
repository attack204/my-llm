#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSpeed 预训练入口（单节点多卡）
已修复：
  1. loss_mask.sum()==0 导致 nan/inf
  2. grad_norm=None 导致 TypeError
  3. lr=0 显示问题（用 scheduler.get_last_lr）
  4. 梯度裁剪默认开启
"""
import os
import time
import math
import argparse
import warnings
import torch
import deepspeed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from k_model import ModelConfig, Transformer
from dataset import PretrainDataset

warnings.filterwarnings("ignore")

# ---------------- 日志 ---------------- #
def log(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)

# ---------------- 初始化模型/分词器 ---------------- #
def init_model_and_tokenizer(lm_config):
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer_k/")
    model = Transformer(lm_config)
    return model, tokenizer

# ---------------- 主函数 ---------------- #
def main():
    parser = argparse.ArgumentParser(description="Tiny-LLM DeepSpeed Pretrain")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed_config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="ckpt_215M")
    parser.add_argument("--data_path", type=str, default="./data.jsonl")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=300)
    args = parser.parse_args()

    # ---------- 1. 分布式 ---------- #
    deepspeed.init_distributed(dist_backend="nccl")
    rank = torch.distributed.get_rank()

    # ---------- 2. 模型 ---------- #
    lm_config = ModelConfig(dim=1024, n_layers=18, max_seq_len=1024, vocab_size=32000)
    model, tokenizer = init_model_and_tokenizer(lm_config)

    # ---------- 3. 数据 ---------- #
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_loader = DataLoader(train_ds,
                              batch_size=16,
                              shuffle=False,
                              num_workers=8,
                              pin_memory=True)

    # ---------- 4. DeepSpeed 引擎 ---------- #
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_ds,
        config=args.deepspeed_config,
    )

    # ---------- 5. 训练参数 ---------- #
    os.makedirs(args.save_dir, exist_ok=True)
    epoch_steps = len(model_engine.training_dataloader)
    total_steps = epoch_steps * args.epochs

    for epoch in range(args.epochs):
        start = time.time()
        for step, batch in enumerate(model_engine.training_dataloader):
            # ---- 数据搬设备 ---- #
            tokens, labels, loss_mask = [t.to(model_engine.device) for t in batch]

            # ---- 前向 ---- #
            outputs = model_engine(tokens, labels)
            loss = outputs.last_loss.view(-1)
            loss_mask = loss_mask.view(-1)
            mask_sum = loss_mask.sum()
            if mask_sum.item() == 0:          # 保护
                if rank == 0:
                    log(f"[WARN] mask_sum==0 at step {step}, skip")
                continue
            loss = (loss * loss_mask).sum() / mask_sum

            # ---- 反向 ---- #
            model_engine.backward(loss)
            model_engine.step()

            # ---- 日志 ---- #
            if step % args.log_interval == 0 and rank == 0:
                lr = scheduler.get_last_lr()[0]              # 真实 lr
                spend = time.time() - start
                remain_min = (spend / (step + 1) * (epoch_steps - step)) / 60
                grad_norm = model_engine.get_global_grad_norm() or 0.0
                log(f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Step [{step}/{epoch_steps}]  "
                    f"loss={loss.item():.3f}  "
                    f"lr={lr:.2e}  "
                    f"grad_norm={grad_norm:.3f}  "
                    f"remain={remain_min:.0f}min")

            # ---- 保存 ---- #
            if (step + 1) % args.save_interval == 0 and rank == 0:
                tag = f"ep{epoch+1}_step{step+1}"
                model_engine.save_checkpoint(args.save_dir, tag=tag)
                log(f"saved checkpoint {tag}")

    # ---------- 6. 最终保存 ---------- #
    if rank == 0:
        model_engine.save_checkpoint(args.save_dir, tag="final")
        log("training done.")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import os
import argparse
import random
from dataclasses import dataclass
from typing import List, Dict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Args:
    model_name: str
    tokenizer_path: str
    dataset_path: str
    dataset_field: str
    out_dir: str
    max_prompt_len: int
    max_gen_len: int
    batch_size: int
    mini_batch_size: int
    ppo_epochs: int
    learning_rate: float
    seed: int
    device: str


def build_reward_fn(keywords: List[str]):
    def reward_fn(prompts: List[str], responses: List[str]) -> List[float]:
        rewards: List[float] = []
        for prompt, resp in zip(prompts, responses):
            score = 0.0
            text = resp.lower()
            # 关键词命中得分
            score += sum(1.0 for k in keywords if k in text)
            # 过短惩罚，过长轻微惩罚
            length = len(resp)
            if length < 10:
                score -= 1.0
            elif length > 512:
                score -= 0.2
            # 非法内容粗略惩罚
            if any(bad in text for bad in ["\tb", "\x00"]):
                score -= 2.0
            rewards.append(float(score))
        return rewards
    return reward_fn


def main():
    parser = argparse.ArgumentParser(description="Tiny-LLM RLHF (PPO) Training")
    parser.add_argument("--model_name", type=str, default="gpt2", help="初始策略模型（HF名称或本地路径）")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer_k", help="分词器路径")
    parser.add_argument("--dataset_path", type=str, default=None, help="包含prompt字段的JSONL/JSON数据集路径")
    parser.add_argument("--dataset_field", type=str, default="prompt", help="数据集中Prompt字段名")
    parser.add_argument("--out_dir", type=str, default="./rl_model", help="模型保存目录")
    parser.add_argument("--max_prompt_len", type=int, default=256)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mini_batch_size", type=int, default=8)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args_ns = parser.parse_args()

    os.makedirs(args_ns.out_dir, exist_ok=True)
    set_seed(args_ns.seed)

    tokenizer = AutoTokenizer.from_pretrained(args_ns.tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|im_end|>"

    # 加载初始策略模型（用于附加ValueHead）
    base_model = AutoModelForCausalLM.from_pretrained(args_ns.model_name)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    model.to(args_ns.device)

    # PPO 配置
    ppo_config = PPOConfig(
        learning_rate=args_ns.learning_rate,
        ppo_epochs=args_ns.ppo_epochs,
        batch_size=args_ns.batch_size,
        mini_batch_size=args_ns.mini_batch_size,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
    )

    # 数据集：需要一列为 prompt
    if args_ns.dataset_path is None:
        # 构造一个极小的演示数据集
        data = {"prompt": [
            "写一条积极的问候语。",
            "给我一个学习建议。",
            "解释什么是强化学习。",
        ]}
        dataset = load_dataset("json", data_files=None, split=None)
        # datasets 不直接支持内存字典，这里临时写入
        tmp_path = os.path.join(args_ns.out_dir, "tmp_prompts.jsonl")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for p in data["prompt"]:
                f.write("{" + f"\"{args_ns.dataset_field}\": \"{p}\"" + "}\n")
        dataset = load_dataset("json", data_files=tmp_path, split="train")
    else:
        dataset = load_dataset("json", data_files=args_ns.dataset_path, split="train")

    def collate_prompts(batch: List[Dict]):
        prompts = [ex[args_ns.dataset_field] for ex in batch]
        inputs = tokenizer(prompts, padding=True, truncation=True, max_length=args_ns.max_prompt_len, return_tensors="pt")
        return prompts, inputs

    # 构建 PPOTrainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=lambda batch: collate_prompts(batch),
    )

    reward_fn = build_reward_fn(keywords=["谢谢", "学习", "强化", "explain", "建议"])  # 可按需调整

    # 训练循环（单轮演示，可按需多轮）
    for step, batch in enumerate(ppo_trainer.dataloader):
        prompts, inputs = batch
        input_ids = inputs["input_ids"].to(args_ns.device)
        attention_mask = inputs["attention_mask"].to(args_ns.device)

        # 生成回复
        with torch.no_grad():
            gen_ids = ppo_trainer.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args_ns.max_gen_len,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 仅保留新生成的部分作为response文本
        responses = []
        for i in range(len(input_ids)):
            prompt_len = input_ids[i].size(0)
            full = gen_ids[i]
            resp_ids = full[prompt_len:]
            responses.append(tokenizer.decode(resp_ids, skip_special_tokens=True))

        # 计算奖励
        rewards = reward_fn(prompts, responses)

        # PPO step 需要将生成的 response ids 传入
        response_tensors = []
        for i in range(len(input_ids)):
            prompt_len = input_ids[i].size(0)
            resp_ids = gen_ids[i][prompt_len:]
            response_tensors.append(resp_ids)

        stats = ppo_trainer.step(input_ids, response_tensors, torch.tensor(rewards, dtype=torch.float32, device=args_ns.device))

        if step % 10 == 0:
            print(f"step {step} rewards: {rewards} avg={sum(rewards)/len(rewards):.3f}")

        if step % 100 == 0 and step > 0:
            save_dir = os.path.join(args_ns.out_dir, f"checkpoint-step{step}")
            os.makedirs(save_dir, exist_ok=True)
            ppo_trainer.model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

    # 最终保存
    ppo_trainer.model.save_pretrained(args_ns.out_dir)
    tokenizer.save_pretrained(args_ns.out_dir)
    print(f"Saved PPO model to {args_ns.out_dir}")


if __name__ == "__main__":
    main()



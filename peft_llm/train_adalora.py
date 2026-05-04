"""
AdaLoRA Training Script
Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning
Based on train_lora_fp16.py; replaces LoraConfig with AdaLoraConfig.
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    AdaLoraConfig,
    get_peft_model,
    TaskType
)
from datasets import load_from_disk
import os
import sys


class Config:
    MODEL_PATH = "/gpool/home/wanghongyang/WangHY/Qlora-llama-2-7B/pre-model/Llama-2-7b-chat-hf"
    FORMATTED_DATA_PATH = "/gpool/home/wanghongyang/WangHY/Qlora-llama-2-7B/data/formatted_data"

    OUTPUT_DIR = "/gpool/home/wanghongyang/WangHY/LightNet/Quant/LoRA/output_adalora"
    CHECKPOINT_DIR = "/gpool/home/wanghongyang/WangHY/LightNet/Quant/LoRA/checkpoints_adalora"
    LOGS_DIR = "/gpool/home/wanghongyang/WangHY/LightNet/Quant/LoRA/logs_adalora"

    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    MAX_LENGTH = 512

    # AdaLoRA config — init_r is starting rank, target_r is pruned rank
    INIT_R = 16
    TARGET_R = 8
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    @staticmethod
    def get_training_args():
        return {
            "num_train_epochs": Config.NUM_EPOCHS,
            "per_device_train_batch_size": Config.BATCH_SIZE,
            "per_device_eval_batch_size": Config.BATCH_SIZE,
            "gradient_accumulation_steps": Config.GRADIENT_ACCUMULATION_STEPS,
            "gradient_checkpointing": True,
            "learning_rate": Config.LEARNING_RATE,
            "fp16": True,
            "save_total_limit": 3,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
            "eval_strategy": "steps",
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "lr_scheduler_type": "cosine",
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "report_to": "tensorboard",
            "remove_unused_columns": False,
            "dataloader_num_workers": 4,
            "group_by_length": True,
            "optim": "adamw_torch",
        }


def setup_training():
    print("=" * 60)
    print("开始 AdaLoRA 训练")
    print(f"模型: {Config.MODEL_PATH}")
    print(f"输出: {Config.OUTPUT_DIR}")
    print(f"Init rank: {Config.INIT_R}, Target rank: {Config.TARGET_R}")
    print("=" * 60)

    if not os.path.exists(Config.MODEL_PATH):
        print(f"错误: 模型路径不存在 {Config.MODEL_PATH}")
        return False
    if not os.path.exists(Config.FORMATTED_DATA_PATH):
        print(f"错误: 数据集不存在 {Config.FORMATTED_DATA_PATH}")
        return False

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOGS_DIR, exist_ok=True)
    print("✓ 所有路径检查通过")
    return True


def main():
    print("开始 AdaLoRA 训练流程...")
    if not setup_training():
        sys.exit(1)

    print("\n1. 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"  ✓ Tokenizer加载成功")

    print("\n2. 加载FP16模型...")
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ 模型加载成功 ({total_params/1e9:.2f}B params)")

    print("\n3. 配置 AdaLoRA...")
    adalora_config = AdaLoraConfig(
        r=Config.TARGET_R,
        target_r=Config.TARGET_R,
        init_r=Config.INIT_R,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=Config.TARGET_MODULES,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, adalora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ AdaLoRA 配置成功")
    print(f"    可训练参数: {trainable_params:,} ({100 * trainable_params / all_params:.4f}%)")

    print("\n4. 加载数据...")
    dataset = load_from_disk(Config.FORMATTED_DATA_PATH)
    print(f"  ✓ 训练集: {len(dataset['train'])}, 验证集: {len(dataset['validation'])}")

    print("\n5. 分词...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=False, max_length=Config.MAX_LENGTH)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    for split in tokenized_dataset.keys():
        if 'text' in tokenized_dataset[split].column_names:
            tokenized_dataset[split] = tokenized_dataset[split].remove_columns(['text'])
    print(f"  ✓ 分词完成")

    print("\n6. 数据整理器...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("\n7. 训练参数...")
    training_args = TrainingArguments(
        output_dir=Config.CHECKPOINT_DIR,
        logging_dir=Config.LOGS_DIR,
        **Config.get_training_args()
    )
    print(f"  ✓ Batch: {Config.BATCH_SIZE}, GradAcc: {Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"  ✓ LR: {Config.LEARNING_RATE}, Epochs: {Config.NUM_EPOCHS}")

    print("\n8. Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    print("\n" + "=" * 60)
    print("开始 AdaLoRA 训练...")
    print("=" * 60)

    trainer.train()

    print("\n保存最终模型...")
    final_path = os.path.join(Config.OUTPUT_DIR, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"模型保存到: {final_path}")
    print("训练完成!")


if __name__ == "__main__":
    main()

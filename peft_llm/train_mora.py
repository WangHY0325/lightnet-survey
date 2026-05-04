"""
MoRA (High-Rank Adaptation) Training Script
Uses higher rank to approximate full-rank updates
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
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import load_from_disk
from config_mora import Config
import os
import sys


def setup_training():
    """设置训练环境"""
    print("=" * 60)
    print("开始MoRA训练 (High-Rank Adaptation)")
    print(f"模型: {Config.MODEL_PATH}")
    print(f"输出: {Config.OUTPUT_DIR}")
    print(f"Rank: {Config.LORA_R} (高秩更新)")
    print("=" * 60)

    # 检查路径
    if not os.path.exists(Config.MODEL_PATH):
        print(f"错误: 模型路径不存在 {Config.MODEL_PATH}")
        return False

    if not os.path.exists(Config.FORMATTED_DATA_PATH):
        print(f"错误: 格式化数据集不存在 {Config.FORMATTED_DATA_PATH}")
        return False

    # 确保输出目录存在
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOGS_DIR, exist_ok=True)

    print("✓ 所有路径检查通过")
    return True


def main():
    print("开始训练流程...")

    # 设置训练环境
    if not setup_training():
        print("初始化失败，退出")
        sys.exit(1)

    print("\n继续执行训练...")

    # 1. 加载tokenizer
    print("\n1. 加载tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)

        # 设置padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("  已设置pad_token为eos_token")

        tokenizer.padding_side = "right"
        print(f"  ✓ Tokenizer加载成功")

    except Exception as e:
        print(f"  ✗ Tokenizer加载失败: {e}")
        return

    # 2. 加载FP16模型
    print("\n2. 加载FP16模型...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        print(f"  ✓ 模型加载成功")
        print(f"    设备映射: {model.hf_device_map}")

        # 打印模型大小
        total_params = sum(p.numel() for p in model.parameters())
        print(f"    总参数量: {total_params:,} ({total_params/1e9:.2f}B)")

    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 配置MoRA (高秩LoRA)
    print("\n3. 配置MoRA适配器 (高秩更新)...")
    try:
        lora_config = LoraConfig(
            r=Config.LORA_R,  # 高秩
            lora_alpha=Config.LORA_ALPHA,
            target_modules=Config.TARGET_MODULES,
            lora_dropout=Config.LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        print(f"  ✓ MoRA配置成功")
        print(f"    Rank: {Config.LORA_R} (高秩)")
        print(f"    目标模块: {Config.TARGET_MODULES}")

        # 打印可训练参数
        trainable_params = 0
        all_params = 0
        for _, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f"    可训练参数: {trainable_params:,}")
        print(f"    总参数: {all_params:,}")
        print(f"    训练比例: {100 * trainable_params / all_params:.4f}%")

    except Exception as e:
        print(f"  ✗ MoRA配置失败: {e}")
        return

    # 4. 加载数据集
    print("\n4. 加载格式化数据集...")
    try:
        dataset = load_from_disk(Config.FORMATTED_DATA_PATH)
        print(f"  ✓ 数据集加载成功")
        print(f"    训练集大小: {len(dataset['train'])}")
        print(f"    验证集大小: {len(dataset['validation'])}")

    except Exception as e:
        print(f"  ✗ 数据集加载失败: {e}")
        return

    # 5. 分词处理
    print("\n5. 分词处理...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=Config.MAX_LENGTH,
        )

    try:
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # 移除原始文本字段
        print("  检查并移除原始文本字段...")
        for split in tokenized_dataset.keys():
            if 'text' in tokenized_dataset[split].column_names:
                print(f"    从{split}分割中移除'text'字段")
                tokenized_dataset[split] = tokenized_dataset[split].remove_columns(['text'])

        print(f"  ✓ 分词完成")

    except Exception as e:
        print(f"  ✗ 分词失败: {e}")
        return

    # 6. 数据整理器
    print("\n6. 配置数据整理器...")
    try:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        print(f"  ✓ 数据整理器配置完成")
    except Exception as e:
        print(f"  ✗ 数据整理器配置失败: {e}")
        return

    # 7. 训练参数
    print("\n7. 配置训练参数...")
    try:
        training_args = TrainingArguments(
            output_dir=Config.CHECKPOINT_DIR,
            logging_dir=Config.LOGS_DIR,
            **Config.get_training_args()
        )
        print(f"  ✓ 训练参数配置完成")

    except Exception as e:
        print(f"  ✗ 训练参数配置失败: {e}")
        return

    # 8. 创建Trainer
    print("\n8. 创建Trainer...")
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
        )
        print(f"  ✓ Trainer创建成功")
    except Exception as e:
        print(f"  ✗ Trainer创建失败: {e}")
        return

    # 9. 开始训练
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    try:
        trainer.train()

        # 10. 保存最终模型
        print("\n10. 保存最终模型...")
        final_model_path = os.path.join(Config.OUTPUT_DIR, "final_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)

        print(f"\n" + "=" * 60)
        print(f"训练完成！")
        print(f"模型保存到: {final_model_path}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n训练被用户中断")
        trainer.save_model(os.path.join(Config.OUTPUT_DIR, "interrupted_model"))

    except Exception as e:
        print(f"  ✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

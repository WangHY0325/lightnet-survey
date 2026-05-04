# config_lora.py - LoRA Baseline Configuration
import torch


class Config:
    # 路径配置
    MODEL_PATH = "/gpool/home/wanghongyang/WangHY/Qlora-llama-2-7B/pre-model/Llama-2-7b-chat-hf"
    FORMATTED_DATA_PATH = "/gpool/home/wanghongyang/WangHY/Qlora-llama-2-7B/data/formatted_data"

    OUTPUT_DIR = "/gpool/home/wanghongyang/WangHY/LightNet/Quant/LoRA/output_lora_fp16"
    CHECKPOINT_DIR = "/gpool/home/wanghongyang/WangHY/LightNet/Quant/LoRA/checkpoints_lora_fp16"
    LOGS_DIR = "/gpool/home/wanghongyang/WangHY/LightNet/Quant/LoRA/logs_lora_fp16"

    # 训练参数
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    MAX_LENGTH = 512

    # LoRA配置 (FP16, no quantization)
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # 不使用量化 (与QLoRA的主要区别)
    USE_QUANTIZATION = False

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
            "optim": "adamw_torch",  # 使用标准AdamW (不是8bit版本)
        }

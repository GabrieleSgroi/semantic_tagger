from peft import LoraConfig
from transformers import BitsAndBytesConfig
import torch


class ModelConfig:
    system_prompt: str = "You are an expert chatbot tagging chunks of texts based on their content. Write a maximum of " \
                         "three short labels representing the texts between square brackets and nothing else. " \
                         "Follow this format [tag1][tag2][tag3]</s>"
    user_prompt: str = """Write labels for the following text: {text}"""
    tags_regex: str = "\[(.*?)\]"
    beta: float = 0.1
    loss_type = "ipo"
    base_model: str = 'HuggingFaceH4/zephyr-7b-alpha'
    answer_split: str = "<|assistant|>\n"
    load_in_4bit: bool = True
    low_cpu_mem_usage: bool = True
    bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(load_in_4bit=True,
                                                        bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_compute_dtype=torch.bfloat16)

    peft_config: LoraConfig = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    max_target_length: int = 16
    max_input: int = 512

    train_params = {
        "gradient_accumulation_steps": 16,
        "gradient_checkpointing": True,
        "num_train_epochs": 1,
        "learning_rate": 1e-5,
        'weight_decay': 0.05,
        "save_total_limit": 10,
        "logging_steps": 1,
        "optim": "paged_adamw_32bit",
        "lr_scheduler_type": 'constant',
        "remove_unused_columns": False,
        "save_strategy": "steps",
        "save_steps": 5,
    }


class EmbeddingConfig:
    model = 'thenlper/gte-large'

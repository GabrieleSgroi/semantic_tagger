import os
from typing import Dict, Callable

from transformers import TrainingArguments, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, \
    PreTrainedTokenizerFast
from trl import DPOTrainer
from datasets import load_dataset, Dataset
from semtag.cfg import ModelConfig


def get_tokens_len(tokenizer: PreTrainedTokenizerFast) -> Callable[[Dict], Dict]:
    def get_len(x: Dict):
        return {'n_tokens': tokenizer(x['prompt'], return_tensors='pt')['input_ids'].squeeze().shape[0]}

    return get_len


def dpo_training(data_path: str,
                 base_model_path: str = ModelConfig.base_model,
                 out_dir: str = './output',
                 checkpoint: str | None = None,
                 batch_size: int = 2,
                 beta: float = ModelConfig.beta,
                 max_prompt_length: int = ModelConfig.max_input,
                 max_target_length: int = ModelConfig.max_target_length,
                 ) -> None:
    """
    Perform preference optimization training with the training argumemnts specified in cfg.ModelConfig.
    Save the model and checkpoint in the specified out_dir.
    This function wraps the DPOTrainer class by Huggingface transformers, see the relevant documentation there.
    Training can be resumed using a checkpoint.
    Args:
        data_path (str): path to the preference dataset.
        base_model_path (str): the base pre-trained model to fine-tune. It can be either a local path or a Hugging Face
                         model ID. It defaults to the parameter set in cfg.ModelConfig.
        out_dir (str): the directory into which the checkpoints and fine-tuned model will be saved.
        checkpoint (str | None): defaults to None. If not None, it will resume training from the provided checkpoint.
        batch_size (int): batch size for training.
        beta (float): the regularization parameter for DPO/IPO training.
                      It defaults to the value set in cfg.ModelConfig.
        max_prompt_length (int): maximum length of the prompt to consider.
                                 It defaults to the value set in cfg.ModelConfig.
        max_target_length (int): maximum number of new tokens to generate.
                                 It defaults to the value set in cfg.ModelConfig.
    """
    training_args = TrainingArguments(output_dir=out_dir,
                                      per_device_train_batch_size=batch_size,
                                      **ModelConfig.train_params)

    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        low_cpu_mem_usage=True,
        quantization_config=ModelConfig.bnb_config,
        device_map="auto",
    )

    ds = load_dataset("csv", data_files=data_path)

    model.config.use_cache = False

    dpo_trainer = DPOTrainer(
        model=model,
        max_length=max_target_length + max_prompt_length + 32,
        max_prompt_length=max_prompt_length,
        args=training_args,
        label_pad_token_id=tokenizer.pad_token_id,
        padding_value=tokenizer.pad_token_id,
        beta=beta,
        truncation_mode='keep_start',
        train_dataset=ds['train'],
        tokenizer=tokenizer,
        loss_type=ModelConfig.loss_type,
        peft_config=ModelConfig.peft_config,
    )
    dpo_trainer.train(resume_from_checkpoint=checkpoint)
    dpo_trainer.save_model(output_dir=out_dir)

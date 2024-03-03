from typing import List

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from semtag.cfg import ModelConfig
import torch
import re
from tqdm import tqdm


class SemanticTagger:
    def __init__(self, adapter_path: str | None):
        """
        Basic class to create tags for text.
        Args:
            adapter_path (str or None): If None the default model specified in cfg/ModelConfig.base_model will be used.
                          If a string, it expects the path to a fine-tuned PEFT adapter.
        """
        self.model = AutoModelForCausalLM.from_pretrained(ModelConfig.base_model,
                                                          low_cpu_mem_usage=ModelConfig.low_cpu_mem_usage,
                                                          quantization_config=ModelConfig.bnb_config,
                                                          device_map="auto",
                                                          )
        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(ModelConfig.base_model)
        self.model.config.use_cache = True

    def tag(self,
            text: str,
            format_tags: bool,
            max_new_tokens: int = ModelConfig.max_target_length,
            **kwargs: object) -> List[str]|str:
        """
        Tag a single text string.
        Args:
            text (str): The text to tag.
            format_tags (bool): if True extract a list of tags, return the unformatted model answer otherwise.
            max_new_tokens (int): Max number of tokens to generate for the answer.
            **kwargs:

        Returns:
            List[str] or str: The generated tags or the unformatted model answer.

        """
        messages = [{"role": "system", "content": ModelConfig.system_prompt},
                    {"role": "user", "content": ModelConfig.user_prompt.format(text=text)}]
        model_inputs = self.tokenizer.apply_chat_template(messages,
                                                          tokenize=True,
                                                          return_tensors='pt',
                                                          add_generation_prompt=True).to('cuda')
        with torch.inference_mode():
            generated_ids = self.model.generate(input_ids=model_inputs,
                                                max_new_tokens=max_new_tokens,
                                                pad_token_id=self.tokenizer.pad_token_id,
                                                **kwargs)
        decoded = self.tokenizer.batch_decode(generated_ids)
        assistant_ans = decoded[0].split(ModelConfig.answer_split)[-1]
        if format_tags:
            tags = re.findall(ModelConfig.tags_regex, assistant_ans)
        else:
            tags = assistant_ans
        return tags

    def batch_tag(self, texts: List[str],
                  batch_size: int,
                  max_new_tokens: int = ModelConfig.max_target_length,
                  **kwargs)->List[str]:
        """
        Generate in batches.
        Args:
            texts (List[str]): List of the texts to tag.
            batch_size (int): The size of the batches to feed to the model.
            max_new_tokens (int): Max number of tokens to generate for the answer.
            **kwargs: keyword arguments to pass to the generate method of the Hugging Face transformer model.

        Returns:
            List[str]: The model generations containing the tags for the input texts.

        """
        n_texts = len(texts)
        n_batches = int(n_texts / batch_size)
        residual = n_texts - n_batches * batch_size
        decoded = []
        with torch.inference_mode():
            for i in tqdm(range(n_batches + 1)):
                if i == n_batches:
                    batch_texts = texts[-residual:]
                else:
                    batch_texts = texts[i * batch_size:(i + 1) * batch_size]
                batch_messages = []
                for txt in batch_texts:
                    msg = [{"role": "system", "content": ModelConfig.system_prompt},
                           {"role": "user", "content": ModelConfig.user_prompt.format(text=txt)}]
                    batch_messages.append(self.tokenizer.apply_chat_template(msg,
                                                                             tokenize=False,
                                                                             add_generation_prompt=True))
                batch_inputs = self.tokenizer(batch_messages, return_tensors='pt', padding='longest').to('cuda')
                generated_ids = self.model.generate(**batch_inputs, max_new_tokens=max_new_tokens, **kwargs)
                decoded.append(self.tokenizer.batch_decode(generated_ids))
        generated_tags = [t.split(ModelConfig.answer_split)[-1] for t in decoded[0]]
        return generated_tags

    def multiple_tags_generation(self, text: str,
                                 num_generations: int,
                                 max_new_tokens: int = ModelConfig.max_target_length,
                                 **kwargs)->List[str]:
        """
        Generate multiple tags for the same text in batches.
        Args:
            text (str): The input text to tag.
            num_generations (int): number of model responses to generate.
            max_new_tokens (int): Max number of tokens to generate for the answer.
            **kwargs: keyword arguments to pass to the generate method of the Hugging Face transformer model.

        Returns:
            List[str]: The model generations containing the tags for the input text.
        """
        messages = [{"role": "system", "content": ModelConfig.system_prompt},
                    {"role": "user", "content": ModelConfig.user_prompt.format(text=text)}]
        model_inputs = self.tokenizer.apply_chat_template(messages,
                                                          tokenize=True,
                                                          return_tensors='pt',
                                                          add_generation_prompt=True).to('cuda')
        model_inputs = model_inputs.repeat(num_generations, 1)
        with torch.inference_mode():
            generated_ids = self.model.generate(input_ids=model_inputs,
                                                max_new_tokens=max_new_tokens,
                                                do_sample=True,
                                                **kwargs)
        decoded = self.tokenizer.batch_decode(generated_ids)
        tags = [x.split(ModelConfig.answer_split)[-1] for x in decoded]
        return tags
from typing import List, Tuple

import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re
import os
from semtag.cfg import ModelConfig
from semtag.similarity import SimilarityScorer


class DataGenerator:
    sys_message = ModelConfig.system_prompt
    user_message = ModelConfig.user_prompt

    def __init__(self,
                 model: str = 'HuggingFaceH4/zephyr-7b-alpha',
                 device_map: str = "auto"):

        self.tokenizer = AutoTokenizer.from_pretrained(model)

        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            low_cpu_mem_usage=ModelConfig.low_cpu_mem_usage,
            quantization_config=ModelConfig.bnb_config,
            device_map=device_map,
        )

    def generate_data(self, texts: List[str]) -> pd.DataFrame:
        data = {"prompt": [], "ans1": [], "tag1": [], "ans2": [], "tag2": [], "original": []}
        for t in tqdm(texts):
            first = None
            messages = [{"role": "system", "content": self.sys_message},
                        {"role": "user", "content": self.user_message.format(text=t)}]
            data["prompt"].append(self.tokenizer.apply_chat_template(messages,
                                                                     tokenize=False,
                                                                     add_generation_prompt=True))
            data['original'].append(t)
            model_inputs = self.tokenizer.apply_chat_template(messages,
                                                              tokenize=True,
                                                              return_tensors='pt',
                                                              add_generation_prompt=True).to('cuda')
            while True:
                generated_ids = self.model.generate(model_inputs,
                                                    max_new_tokens=ModelConfig.max_target_length,
                                                    do_sample=True,
                                                    )
                decoded = self.tokenizer.batch_decode(generated_ids)
                assistant_ans = decoded[0].split(ModelConfig.answer_split)[-1]
                extracted_tags = re.findall(ModelConfig.tags_regex, assistant_ans)
                if len(extracted_tags) == 0:
                    continue
                else:
                    if first is None:
                        data["ans1"].append(assistant_ans)
                        data["tag1"].append(', '.join(extracted_tags))
                        first = set(extracted_tags)
                    else:
                        if set(extracted_tags) != first:
                            data["ans2"].append(assistant_ans)
                            data["tag2"].append(', '.join(extracted_tags))
                            break
        return pd.DataFrame.from_dict(data)


def reconstruct_from_tags(x: str) -> str:
    tags = re.findall("\[(.*?)\]", x)
    if len(tags) == 0:
        ans = None
    else:
        # cap number of tags
        upper = min(len(tags), 3)
        tags = tags[:upper]
        ans = ''
        for t in tags:
            ans += f'[{t}]'
        ans = ans + '</s>'
    return ans


def has_correct_format(x: str) -> bool:
    x = x.replace(' ', '')  # remove whitespace
    x = x.replace('</s>', '')  # remove end token
    tags = re.findall("\[(.*?)\]", x)
    correct_format = ''
    for tag in tags:
        correct_format += f'[{tag}]'
    n_tags = len(tags)
    valid = False
    if (n_tags < 4) and (n_tags > 0) and x == correct_format:
        valid = True
    return valid


def format_generated_dataset(data_path: str) -> pd.DataFrame:
    generated_data = []
    for file in os.listdir(data_path):
        if not file.endswith('.csv'):
            continue
        generated_data.append(pd.read_csv(os.path.join(data_path, file)))
    generated_data = pd.concat(generated_data)
    # Extra check for duplicates
    generated_data = generated_data.drop_duplicates(subset='original')
    generated_data['is_ans1_valid'] = generated_data['ans1'].apply(has_correct_format)
    generated_data['is_ans2_valid'] = generated_data['ans2'].apply(has_correct_format)
    return generated_data


def fix_typo(df: pd.DataFrame) -> pd.DataFrame:
    df['prompt'] = df['prompt'].apply(lambda x: re.sub('ofthree', 'of three', x))
    return df


def remove_long_inputs(df: pd.DataFrame, model_path: str, len_treshold: int = 512) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    l = []
    for i, row in df.iterrows():
        tokens = tokenizer(row['prompt'], return_tensors='pt')['input_ids']
        l.append(tokens.squeeze().shape[0])
    long_flag = np.array(l) > len_treshold
    print(f'Removed {long_flag.sum()} elements that were too long')
    return df[~long_flag]


def select_by_format(df: pd.DataFrame) -> pd.DataFrame:
    preferred = {'prompt': df['prompt'].to_list(), "chosen": [], "rejected": []}
    for i, row in df.iterrows():
        if row['is_ans1_valid'] and (not row['is_ans2_valid']):
            preferred['chosen'].append(row['ans1'])
            preferred['rejected'].append(row['ans2'])
        elif row['is_ans2_valid'] and (not row['is_ans1_valid']):
            preferred['chosen'].append(row['ans2'])
            preferred['rejected'].append(row['ans1'])
        else:
            raise RuntimeError(
                f"Selection by format works only when exactly one of the two answer is not valid. Found ans1 valid "
                f"{row['is_ans1_valid']} and ans2 valid {row['is_ans2_valid']}")
    return pd.DataFrame.from_dict(preferred)


def rebalance_data(similarity_preference: pd.DataFrame,
                   format_preference: pd.DataFrame,
                   discarded: pd.DataFrame,
                   format_choice_ratio: float = 0.5
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rebalance texts so that texts selected by semantic similarity is (approximately) equal to format_choice_ratio
     times the total amount of texts"""
    n_similarity_choice = int(len(format_preference) * (1 / format_choice_ratio - 1))
    # this is an approximation, not all texts can be recovered
    if len(similarity_preference) > n_similarity_choice:
        similarity_preference = similarity_preference.iloc[:n_similarity_choice]
    else:
        n_to_sanitize = n_similarity_choice - len(similarity_preference)
        if len(discarded) > n_to_sanitize:
            sanitized = discarded.sample(n_to_sanitize)
        else:
            n_to_sanitize = len(format_preference) - int(format_choice_ratio * (len(similarity_preference)
                                                                                + len(discarded)
                                                                                + len(format_preference)))
            sanitized = format_preference.sample(n_to_sanitize)
            sanitized = pd.concat([sanitized, discarded])
        format_preference = format_preference[~format_preference['prompt'].isin(sanitized['prompt'])]
        sanitized['ans1'] = sanitized['ans1'].apply(reconstruct_from_tags)
        sanitized['ans2'] = sanitized['ans2'].apply(reconstruct_from_tags)
        sanitized = sanitized.dropna(subset=['ans1', 'ans2'])
        print('Number of elements recovered: ', len(sanitized))
        similarity_preference = pd.concat([similarity_preference, sanitized])
    return similarity_preference, format_preference


def create_preference_data(data_path: str, max_input_length: int = 500,
                           format_choice_ratio: float | None = 0.5) -> pd.DataFrame:
    df = format_generated_dataset(data_path)
    df = fix_typo(df)
    df = remove_long_inputs(df, model_path=ModelConfig.base_model, len_treshold=max_input_length)
    # remove elements in which both answers have the wrong format
    discarded = df[~(df['is_ans1_valid'] | df['is_ans2_valid'])]
    print('Elements in which both answers have the wrong format:', len(discarded))
    df = df[(df['is_ans1_valid'] | df['is_ans2_valid'])]
    # score by similarity if both have the correct format
    to_score = df[(df['is_ans1_valid'] & df['is_ans2_valid'])]
    format_choice = df[~df['prompt'].isin(to_score['prompt'])]
    if format_choice_ratio is not None:
        to_score, format_choice = rebalance_data(similarity_preference=to_score,
                                                 format_preference=format_choice,
                                                 discarded=discarded,
                                                 format_choice_ratio=format_choice_ratio)
    format_selected = select_by_format(format_choice)
    print('Data to score by similarity: ', len(to_score))
    print('Data selected by format: ', len(format_selected))
    scorer = SimilarityScorer()
    scored = scorer.create_preference_dataset(generated_data=to_score)
    return pd.concat([scored, format_selected]).sample(frac=1)


if __name__ == '__main__':
    generated_path = './generated_tags'
    preference_data = create_preference_data(generated_path, format_choice_ratio=0.2)
    preference_data.to_csv(os.path.join('./preference.csv'), index=False)

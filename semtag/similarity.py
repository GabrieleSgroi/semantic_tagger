import warnings
from typing import List

import pandas as pd
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from semtag.cfg import ModelConfig, EmbeddingConfig
import numpy as np

class SimilarityScorer:
    def __init__(self, model: str = EmbeddingConfig.model):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)

    def get_split_embeddings(self, text: str, max_tokens: int, device: str) -> Tensor:
        self.model.to(device)
        encoded_input = self.tokenizer(text, padding=False, truncation=False, return_tensors='pt')
        n_splits = int((encoded_input['input_ids']).shape[1] / max_tokens)
        if n_splits > 0:
            encoded_splits = [encoded_input['input_ids'][:, i * max_tokens:(i + 1) * max_tokens] for i in
                              range(n_splits)]
            encoded_splits = torch.cat(encoded_splits, dim=0)
            with torch.inference_mode():
                model_output = self.model(encoded_splits.to(device))
            embs = F.normalize(model_output[0].mean(dim=1), p=2, dim=1)
        else:
            embs = None
        if encoded_input['input_ids'].shape[1] > max_tokens * n_splits:
            residual_encoded = encoded_input['input_ids'][:, max_tokens * n_splits:]
            with torch.inference_mode():
                residual_output = self.model(residual_encoded.to(device))
            residual_emb = F.normalize(residual_output[0].mean(dim=1), p=2, dim=1)
            if embs is None:
                embs = residual_emb
            else:
                embs = torch.cat([embs, residual_emb], dim=0)
        return embs

    def get_batch_embeddings(self, texts: List[str], device: str) -> Tensor:
        self.model.to(device)
        encoded_input = self.tokenizer(texts,
                                       padding='max_length',
                                       truncation=True,
                                       max_length=ModelConfig.max_input,
                                       return_tensors='pt').to(device)
        with torch.inference_mode():
            model_output = self.model(**encoded_input)[0]
        embs = F.normalize(model_output.mean(dim=1), p=2, dim=1)
        return embs

    def get_split_similarity(self, tags: str, reference: str, max_tokens: int, device: str) -> Tensor:
        tag_embs = self.get_split_embeddings(tags, max_tokens=max_tokens, device=device)
        ref_embs = self.get_split_embeddings(reference, max_tokens=max_tokens, device=device)
        if len(tag_embs) > 1:
            warnings.warn(f"Text:\n{tags} has more than the set {max_tokens} max tokens, considering only the "
                          f"{max_tokens} tokens to compare similarity ")
            tag_embs = tag_embs[0].unsqueeze(0)

        # Take the max of the similarity with the splits of the reference texts if the latter surpasses the max tokens
        cos_sim = torch.einsum('bi, bi->b', tag_embs, ref_embs).max().item()
        return cos_sim

    def get_similarity_batched(self, tags: pd.Series, reference: pd.Series, batch_size: int, device: str) -> Tensor:
        n_batches = int(len(tags) / batch_size)
        residue = len(tags) - n_batches * batch_size
        tags = list(tags)
        reference = list(reference)
        cos_sim = []
        for i in range(n_batches):
            tags_batch = self.get_batch_embeddings(tags[i * batch_size:(i + 1) * batch_size], device=device)
            refs_batch = self.get_batch_embeddings(reference[i * batch_size:(i + 1) * batch_size], device=device)
            cos_sim.append(torch.einsum('bi,bi->b',
                                        tags_batch,
                                        refs_batch))
        if residue > 0:
            tags_batch = self.get_batch_embeddings(tags[-residue:], device=device)
            residue_batch = self.get_batch_embeddings(reference[-residue:], device=device)
            cos_sim.append(torch.einsum('bi,bi->b',
                                        tags_batch,
                                        residue_batch))
        cos_sim = torch.cat(cos_sim)
        return cos_sim

    def create_preference_dataset(self,
                                  generated_data: pd.DataFrame,
                                  max_tokens: int = 512,
                                  device: str = 'cuda') -> pd.DataFrame:
        # THIS IS INEFFICIENT, FOR LARGE DATASETS IT WOULD BE BETTER TO COMPUTE ALL THE EMBEDDINGS IN BATCH BEFOREHAND
        preference_data = {"prompt": generated_data["prompt"].to_list(), "chosen": [], "rejected": []}
        for i, row in tqdm(generated_data.iterrows(), total=generated_data.shape[0]):
            sim1 = self.get_split_similarity(tags=row['tag1'],
                                             reference=row['original'],
                                             max_tokens=max_tokens,
                                             device=device)
            sim2 = self.get_split_similarity(tags=row['tag2'],
                                             reference=row['original'],
                                             max_tokens=max_tokens,
                                             device=device)
            if sim1 >= sim2:
                preference_data["chosen"].append(row['ans1'])
                preference_data["rejected"].append(row['ans2'])
            else:
                preference_data["chosen"].append(row["ans2"])
                preference_data["rejected"].append(row["ans1"])

        return pd.DataFrame.from_dict(preference_data)

    def embed(self, texts: pd.Series, batch_size: int, device: str)->np.ndarray:
        texts = texts.tolist()
        n_batches = int(len(texts) / batch_size)
        residual = len(texts) - n_batches * batch_size
        embs = []
        for i in range(n_batches):
            embs.append(self.get_batch_embeddings(texts[i + n_batches:(i + 1) * n_batches],
                                                  device=device).cpu())
        if residual > 0:
            embs.append(self.get_batch_embeddings(texts[-residual:], device=device).cpu())
        embs = torch.cat(embs, dim=0)
        return embs.numpy()


if __name__ == '__main__':
    from tqdm import tqdm

    df = pd.read_csv(
        'C:\\Users\\Gabry\\PycharmProjects\\semantic_tagger\\data\\generated_tags\\merged_sanitized.csv')
    scorer = SimilarityScorer()
    preference_df = scorer.create_preference_dataset(generated_data=df)
    preference_df.to_csv("C:\\Users\\Gabry\\PycharmProjects\\semantic_tagger\\texts\\preference.csv", index=False)
    print(preference_df)

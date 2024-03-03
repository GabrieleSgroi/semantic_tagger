import argparse
import re

import pandas as pd

from semtag.cfg import ModelConfig
from semtag.data import has_correct_format
import torch
import gc
from semtag.similarity import SimilarityScorer
import numpy as np


def test_format(test_df: pd.DataFrame,
                tags_column: str,
                format_column: str) -> None:
    test_df[format_column] = test_df[tags_column].apply(has_correct_format)


def test_semantic_similarity(test_df: pd.DataFrame,
                             tags_column: str,
                             text_column: str,
                             similarity_column: str,
                             batch_size: int,
                             extract_tags: bool,
                             device: str = 'cuda') -> np.ndarray:
    scorer = SimilarityScorer()
    if extract_tags:
        tags = test_df[tags_column].apply(lambda x: ','.join(re.findall(ModelConfig.tags_regex, x)))
    else:
        tags = test_df[tags_column]
    sim = scorer.get_similarity_batched(tags=tags,
                                        reference=test_df[text_column],
                                        batch_size=batch_size,
                                        device=device,
                                        )
    test_df[similarity_column] = sim.cpu().numpy()
    gc.collect()
    torch.cuda.empty_cache()
    return sim.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path')
    parser.add_argument('-p', '--output_path', default=None)
    parser.add_argument('-i', '--text_column', default='context')
    parser.add_argument('-o', '--tag_column', default='tags')
    parser.add_argument('-f', '--format_column', default='correct_format')
    parser.add_argument('-s', '--similarity_column', default='semantic_similarity')
    parser.add_argument('-e', '--extract_tags', default='True')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    args = parser.parse_args()

    if args.output_path is None:
        output_path = args.data_path
    else:
        output_path = args.output_path

    if args.extract_tags == 'True':
        extract = True
    elif args.extract_tags == 'False':
        extract = False
    else:
        raise ValueError(f"Argument 'extract_tags' can only be 'True' or 'False'. Got {args.extract_tags}")

    tagged_df = pd.read_csv(args.data_path)
    test_format(test_df=tagged_df,
                tags_column=args.tag_column,
                format_column=args.format_column)
    correct = tagged_df[args.format_column].sum()
    print('Tagger correct format ratio', correct / len(tagged_df))
    similarities = test_semantic_similarity(test_df=tagged_df,
                                            tags_column=args.tag_column,
                                            text_column=args.text_column,
                                            similarity_column=args.similarity_column,
                                            batch_size=args.batch_size,
                                            extract_tags=extract)
    std = similarities.std()
    print(f'Similarity_mean: {similarities.mean()}+-{std / np.sqrt(len(similarities))}')
    print('Similarity std', std)
    tagged_df.to_csv(args.output_path, index=False)

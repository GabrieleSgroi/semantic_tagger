import argparse

import pandas as pd

from semtag.cfg import ModelConfig
from semtag.tagger import SemanticTagger
from time import time


def get_tags(df: pd.DataFrame,
             text_column: str,
             tags_col_name: str,
             adapter_path: str | None,
             sample: bool,
             format_tags: bool,
             max_new_tokens: int = ModelConfig.max_target_length,
             **kwargs
             ) -> pd.DataFrame:
    start = time()
    print('Creating tags')
    model = SemanticTagger(adapter_path=adapter_path)
    if format_tags:
        df[tags_col_name] = df[text_column].apply(lambda x: ', '.join(model.tag(text=x,
                                                                           max_new_tokens=max_new_tokens,
                                                                           do_sample=sample,
                                                                           format_tags=format_tags,
                                                                           **kwargs)))
    else:
        df[tags_col_name] = df[text_column].apply(lambda x: model.tag(text=x,
                                                                      max_new_tokens=max_new_tokens,
                                                                      do_sample=sample,
                                                                      format_tags=format_tags,
                                                                      **kwargs))
    print('inference time', time() - start)
    return df


if __name__ == '__main__':
    description = "Create tags for texts. The input texts must be contained in a column of a csv file whose name is " \
                  "specified by the `text_column` arguments. A csv containing another column with the tags will" \
                  "be saved. Passing `output_path` equal to `data_path` will overwrite the input csv adding the tag " \
                  "column."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_path', help="Path to the csv file containing the texts.")
    parser.add_argument('-p', '--output_path', default='./tagged.csv', help="Path to the generated"
                                                                            " csv containing the tags. "
                                                                            "(Default: './tagged.csv')")
    parser.add_argument('-i', '--text_column', default='context',
                        help="The name of the column containing the input texts. (Default: 'context')")
    parser.add_argument('-o', '--tag_column', default='tags', help="The name of the column in which the tags will be "
                                                                   "saved. (Default: `tags`)")
    parser.add_argument('-t', '--temperature', type=float, default=0., help="Temperature of the generation sampling."
                                                                            "0 means deterministic sampling. "
                                                                            "(Default: 0)")
    parser.add_argument('-c', '--checkpoint', default=None, help="Path to the fine-tuned adapter. If None the "
                                                                 "default pre-trained model specified in "
                                                                 "semtag.cfg.ModelConfig will be used. (Default: None)")
    parser.add_argument('-l', '--max_new_tokens', type=int, default=32, help="Max number of tokens to generate. "
                                                                             "(Default: 32)")
    parser.add_argument('-f', '--format_tags', default='True', help="Wheter to format the model answer. It can be "
                                                                    "either 'True' or 'False'. If 'True' the tags will "
                                                                    "be extracted from the model answer and separated "
                                                                    "by commas. (Default: 'True')")
    args = parser.parse_args()
    if args.temperature > 0:
        sample = True
        temperature = args.temperature
    else:
        sample = False
        temperature = None
    if args.format_tags == 'True':
        format_tags = True
    elif args.format_tags == 'False':
        format_tags = False
    else:
        raise RuntimeError(f"Argument 'format_tags' must be either 'True' or 'False', got '{args.format_tags}'")
    df = pd.read_csv(args.data_path)
    df = get_tags(df=df,
                  text_column=args.text_column,
                  tags_col_name=args.tag_column,
                  sample=sample,
                  temperature=temperature,
                  max_new_tokens=args.max_new_tokens,
                  adapter_path=args.checkpoint,
                  format_tags=format_tags,
                  )
    df.to_csv(args.output_path, index=False)

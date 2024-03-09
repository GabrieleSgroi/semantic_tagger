# Semantic Tagging: Create Meaningful Tags for your Text Data 
This code is meant for educational purposes only. <br>
This code accompanies the blog post [Semantic Tagging: Create Meaningful Tags for your Text Data](https://medium.com/p/dcf8d2f24960) <br>
The default settings use [zephyr-7b-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha) as the generative model
and [gte-large](https://huggingface.co/thenlper/gte-large) as the embedding model. Please check the 
linked model cards for more information and licenses.

## Usage 
### Installation
```shell
git clone https://github.com/GabrieleSgroi/semantic_tagger.git
pip install ./semantic_tagger
 ```
### Generate tags
````shell
usage: python3 scripts/tag_data.py [-h] [-d DATA_PATH] [-p OUTPUT_PATH] [-i TEXT_COLUMN] [-o TAG_COLUMN] [-t TEMPERATURE]
                   [-c CHECKPOINT] [-l MAX_NEW_TOKENS] [-f FORMAT_TAGS]

Create tags for texts. The input texts must be contained in a column of a csv file whose name is specified by the
`text_column` arguments. A csv containing another column with the tags willbe saved. Passing `output_path` equal to
`data_path` will overwrite the input csv adding the tag column.

options:
  -h, --help            show this help message and exit
  -d DATA_PATH, --data_path DATA_PATH
                        Path to the csv file containing the texts.
  -p OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to the generated csv containing the tags. (Default: './tagged.csv')
  -i TEXT_COLUMN, --text_column TEXT_COLUMN
                        The name of the column containing the input texts. (Default: 'context')
  -o TAG_COLUMN, --tag_column TAG_COLUMN
                        The name of the column in which the tags will be saved. (Default: `tags`)
  -t TEMPERATURE, --temperature TEMPERATURE
                        Temperature of the generation sampling.0 means deterministic sampling. (Default: 0)
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        Path to the fine-tuned adapted. If None the default pre-trained model specified in
                        semtag.cfg.ModelCOnfig will be used. (Default: None)
  -l MAX_NEW_TOKENS, --max_new_tokens MAX_NEW_TOKENS
                        Max number of tokens to generate. (Default: 32)
  -f FORMAT_TAGS, --format_tags FORMAT_TAGS
                        Wheter to format the model answer. It can be either 'True' or 'False'. If 'True' the tags will
                        be extracted from the model answer and separated by commas. (Default: 'True')
````
### Fine-tune on preference tags
```shell
usage: python3 scripts/finetune.py [-h] [-d DATA_PATH] [-b BATCH_SIZE] [-c CHECKPOINT] [-o SAVE_DIR]

Fine-tune the model using the parameters specified in semtag.cfg.ModelConfig. Save the fine-tuned model and the
checkpoints in the directory specified by the `save_dir` argument.

options:
  -h, --help            show this help message and exit
  -d DATA_PATH, --data_path DATA_PATH
                        Path to the preference dataset. See the documentation of the HuggingFace DPOTrainer class for
                        the required format: https://huggingface.co/docs/trl/main/en/dpo_trainer
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size to use for training.
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        Path to a checkpoint to resume training. If None a new training instance will be started.
                        (Default: None).
  -o SAVE_DIR, --save_dir SAVE_DIR
                        Path to the directory into which the checkpointsand fine-tuned model will be saved. (Default:'./output')
```

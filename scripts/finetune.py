import argparse
from semtag.cfg import ModelConfig
from semtag.dpo_training import dpo_training

if __name__ == '__main__':
    description = "Fine-tune the model using the parameters specified in semtag.cfg.ModelConfig. Save the fine-tuned " \
                  "model and the checkpoints in the directory specified by the `save_dir` argument."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_path', help="Path to the preference dataset. See the documentation of the Hugging"
                                                  "Face DPOTrainer class for the required format:"
                                                  " https://huggingface.co/docs/trl/main/en/dpo_trainer")
    parser.add_argument('-b', '--batch_size', type=int, help="batch size to use for training.")
    parser.add_argument('-c', '--checkpoint', default=None, help="Path to a checkpoint to resume training. If None a "
                                                                 "new training instance will be started. "
                                                                 "(Default: None).")
    parser.add_argument('-o', '--save_dir', default='./output', help="Path to the directory into which the checkpoints"
                                                                     "and fine-tuned model will be saved. "
                                                                     "(Default: './output')")
    args = parser.parse_args()
    dpo_training(data_path=args.data_path,
                 base_model_path=ModelConfig.base_model,
                 out_dir=args.save_dir,
                 checkpoint=args.checkpoint,
                 batch_size=args.batch_size,
                 beta=ModelConfig.beta,
                 max_prompt_length=ModelConfig.max_input,
                 max_target_length=ModelConfig.max_target_length)
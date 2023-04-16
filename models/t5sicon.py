from simplet5 import SimpleT5
import torch

torch.set_float32_matmul_precision('medium')

def train(train_df, eval_df, prototype="t5", base_model="t5-large", logger="default",
          args={}):
    # instantiate
    model = SimpleT5()
    # load (supports t5, mt5, byT5 models)
    model.from_pretrained(prototype, base_model)
    # train
    model.train(train_df=train_df, # pandas dataframe with 2 columns: source_text & target_text
                eval_df=eval_df, # pandas dataframe with 2 columns: source_text & target_text
                source_max_token_len = args.get("source_max_token_len", 512), 
                target_max_token_len = args.get("target_max_token_len", 512),
                batch_size = args.get("batch_size", 8),
                max_epochs = args.get("max_epochs", 3),
                use_gpu = args.get("use_gpu", 3),
                outputdir = args.get("output_dir", "outputs"),
                early_stopping_patience_epochs = 0,
                precision = 16,
                dataloader_num_workers=32,
                logger=logger
    )

    model.model.save_pretrained(args.get("output_dir", "outputs"))
    model.tokenizer.save_pretrained(args.get("output_dir", "outputs"))


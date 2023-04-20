from simplet5 import SimpleT5
import torch

#torch.set_float32_matmul_precision('medium')

def train(train_df, eval_df, prototype="t5", base_model="t5-large", logger="default",
          args={}):
    # instantiate
    model = SimpleT5()
    # load (supports t5, mt5, byT5 models)
    model.from_pretrained(prototype, base_model)
    # train
    train_df = train_df[~train_df["source_text"].contains("[deleted]")]
    eval_df = eval_df[~eval_df["source_text"].contains("[deleted]")]
    model.train(train_df=train_df, # pandas dataframe with 2 columns: source_text & target_text
                eval_df=eval_df, # pandas dataframe with 2 columns: source_text & target_text
                source_max_token_len = args.get("source_max_token_len", 512), 
                target_max_token_len = args.get("target_max_token_len", 512),
                batch_size = args.get("batch_size", 2),
                max_epochs = args.get("max_epochs", 3),
                use_gpu = args.get("use_gpu", 3),
                outputdir = args.get("output_dir", "outputs"),
                early_stopping_patience_epochs = args.get("early_stopping_patience_epochs", 3),
                precision = args.get("precision", 32),
                dataloader_num_workers= args.get("dataloader_num_workers", 32),
                save_only_last_epoch= args.get("save_only_last_epoch", True),
                logger=logger,
                strategy="ddp"
    )

    model.model.save_pretrained(args.get("output_dir", "outputs")+"_model")
    model.tokenizer.save_pretrained(args.get("output_dir", "outputs")+"_model")


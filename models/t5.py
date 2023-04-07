# import
from simplet5 import SimpleT5
from data import load_as_df
import torch

torch.set_float32_matmul_precision('medium')

# instantiate
model = SimpleT5()
train_df, eval_df, val = load_as_df()

def train(prototype="t5", base_model="../t5-large", output_dir="outputs"):
    # load (supports t5, mt5, byT5 models)
    model.from_pretrained(prototype, base_model)


    #small_train = train_df[:int(len(train_df)*0.1)]
    # train
    model.train(train_df=train_df, # pandas dataframe with 2 columns: source_text & target_text
                eval_df=eval_df, # pandas dataframe with 2 columns: source_text & target_text
                source_max_token_len = 512, 
                target_max_token_len = 512,
                batch_size = 8,
                max_epochs = 3,
                use_gpu = 3,
                outputdir = output_dir,
                early_stopping_patience_epochs = 0,
                precision = 16,
                dataloader_num_workers=32
                )

    model.model.save_pretrained(output_dir)
    model.tokenizer.save_pretrained(output_dir)

def test(prototype="t5", model_path="outputs"):
    # load trained T5 model
    model.load_model(prototype, model_path, use_gpu=False)

    # predict
    for s in val:
        print(model.predict(s["source_text"]))
        print(s["target_text"])
        print("-----\t\t")

def both(prototpype="t5", base_model="../t5-large", output_dir="outputs"):
    train(prototype, base_model, output_dir)
    test(prototype, output_dir)

if __name__ == "__main__":
    test()
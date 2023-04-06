# import
from simplet5 import SimpleT5
from data import load_as_df

# instantiate
model = SimpleT5()

# load (supports t5, mt5, byT5 models)
model.from_pretrained("t5","../t5-large")

train_df, eval_df, val = load_as_df()

# train
model.train(train_df=train_df, # pandas dataframe with 2 columns: source_text & target_text
            eval_df=eval_df, # pandas dataframe with 2 columns: source_text & target_text
            source_max_token_len = 512, 
            target_max_token_len = 512,
            batch_size = 4,
            max_epochs = 5,
            use_gpu = 3,
            outputdir = "outputs",
            early_stopping_patience_epochs = 0,
            precision = 32,
            )

# load trained T5 model
model.load_model("t5-prochoice", "outputs", use_gpu=False)

# predict
for s, t in val.itertuples(index=False):
    print(model.predict(s))
    print(t)
    print("-----\t\t")
# 3. Load wandb
# 1. Load data using templates
# 2. split in to train and test
# ----
# 4. train
# 5. upload model to wandb

""" Data Processing
{
    "version": "v0",
    "target": "prochoice_PCTS",
    "description": "Prochoice dataset with parent child toxicity summary",
    "wandb-project": "knoxcs/detoxify",
    "base-dataset": "prochoice.summarized.json:v0",
    "dataset": {
        "path": "./artifacts/detoxify/prochoice.summarized.json:v0/prochoice.summarized.json",
        "preprocess": "get_parent_child_toxic_summary"
    },
    "split": {
        "train": 0.5,
        "eval": 0.2
    }
}
"""

import wandb, json
from templates.templates import process_data
from sklearn.model_selection import train_test_split
import pandas as pd
from models import t5sicon

def get_dataset(config):
    ds = wandb.use_artifact(config["base-dataset"], type="dataset")
    ds.download()
    return process_data(config["dataset"])

def split_dataset(config, ds):
    train_frac = config["split"]["train"]
    eval_frac = config["split"]["eval"]
    create_split(train_frac, eval_frac, ds)

def create_split(ds, train_frac, eval_frac):
    if type(ds) == str:
        ds = pd.read_json(ds)
    train, test = train_test_split(ds, train_size=train_frac, random_state=42, shuffle=True)
    eval_frac = eval_frac/(1-train_frac)
    eval_df, test = train_test_split(test, train_size=eval_frac, random_state=42, shuffle=True)
    name = "{}.json"
    train.to_json("split/" + name.format("train"))
    eval_df.to_json("split/" + name.format("eval"))
    test.to_json("split/" + name.format("test"))

def load_split(config):
    train = pd.read_json("train.json")
    eval_df = pd.read_json("eval.json")
    test = pd.read_json("test.json")
    return train, eval_df, test

def build_t5_dataset(config):
    print("running config:", config)
    config = json.load(open(config))
    entity, project = config["wandb-project"].split("/")
    wandb_logger = wandb.init(project=project, entity=entity, config=config)
    ds = get_dataset(config)
    split_dataset(config, ds)
    artifact = wandb.Artifact(config["target"], type="dataset", description=config["description"])
    artifact.add_file("train.json")
    artifact.add_file("test.json")
    artifact.add_file("eval.json")
    wandb_logger.log_artifact(artifact)
    
from pytorch_lightning.loggers import WandbLogger
# def train(train_df, eval_df, prototype="t5", base_model="t5-large", output_dir="outputs", logger="default"):
def train_t5(config):
    config = json.load(open(config))
    entity, project = config["wandb-project"].split("/")

    experiment = wandb.init(project=project, entity=entity, group="hyperion")
    dataset = wandb.use_artifact(config["wandb-project"] + "/" + config["dataset"])
    dataset = dataset.download()
    train = pd.read_json(dataset+"/train.json")
    eval_df = pd.read_json(dataset+"/eval.json")
    wandb_logger = WandbLogger(name=config["name"], experiment=experiment, log_model=False, project=project, entity=entity, config=config, group="hyperion", tags=["model", config["prototype"], config["base_model"]])
    print("starting training")
    t5sicon.train(train, 
                eval_df, 
                prototype=config["prototype"], 
                base_model=config["base_model"], 
                logger=wandb_logger,
                args=config.get("args", {}))
    artifact = wandb.Artifact(config["name"], type="model")
    artifact.add_dir(config["args"]["output_dir"]+"_model")
    experiment.log_artifact(artifact)

"""
{
    "dataset": "prochoice_PCTS:v0",
    "wandb-project": "knoxcs/detoxify",
    "prototype": "t5",
    "base_model": "t5-large",
    "args": {
        "source_max_token_len": 512,
        "target_max_token_len": 512,
        "batch_size": 8,
        "max_epochs": 3,
        "use_gpu": 3,
        "output_dir": "outputs",
    }
}
"""

if __name__ == "__main__":
    import os
    r = os.environ["T5RUN"]
    train_t5(r)
    
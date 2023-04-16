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

def get_dataset(config):
    ds = wandb.use_artifact(config["base-dataset"], type="dataset")
    ds.download()
    return process_data(config["dataset"])

def split_dataset(config, ds):
    train_frac = config["split"]["train"]
    train, test = train_test_split(ds, train_size=train_frac, random_state=42, shuffle=True)
    eval_frac = config["split"]["eval"]
    eval_frac = eval_frac/(1-train_frac)
    eval_df, test = train_test_split(test, train_size=eval_frac, random_state=42, shuffle=True)
    name = "{}.json"
    train.to_json(name.format("train"))
    eval_df.to_json(name.format("eval"))
    test.to_json(name.format("test"))

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
    
# def train(train_df, eval_df, prototype="t5", base_model="t5-large", output_dir="outputs", logger="default"):
def run_t5(config):
    config = json.load(open(config))
    dataset = wandb.use_artifact(config["project"] + "/" + config["dataset"])
    entity, project = config["wandb-project"].split("/")
    wandb_logger = wandb.init(project=project, entity=entity, config=config)
    train = pd.read_json(dataset+"/train.json")
    eval = pd.read_json(dataset+"/eval.json")
    sicon.train(train, 
                evaldf, 
                config["prototype"], 
                config["base_model"], 
                config["output_dir"], 
                config["epochs"], 
                wandb_logger,
                config.get("args", {}))

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
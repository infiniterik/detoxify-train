from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GPT2Tokenizer, GPTNeoForCausalLM#, Trainer, TrainingArguments
from optimum.onnxruntime import ORTTrainer, ORTTrainingArguments
from torch.utils.data import Dataset
from optimum.pipelines import pipeline
from datasets import load_from_disk
import wandb
import torch
from tqdm import tqdm

def load_model(model_name, tokenizer_name=None, use_onnx=False):
    # Download pretrained GPT-NEO model and tokenizer
    # load tokenizer using tokenizer_name if it exists, otherwise use model_name
    if not tokenizer_name:
        tokenizer_name = model_name
    if "t5" in tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name) if tokenizer_name else GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token="<|pad|>"
    tokenizer.bos_token="<|startoftext|>"
    tokenizer.eos_token="<|endoftext|>"
    if use_onnx:
        #model = pipeline("text-generation", model=model_name, accelerator="ort")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, accelerator="ort")
    else:
        model = GPTNeoForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def make_non_toxic(text):
    # if the string contains "\n\nA toxic reply:" then replace it with "\n\nA non-toxic reply:" and truncate the string at the first "\n\nA non-toxic reply:"
    toxic = "\n\nA toxic reply: "
    nontoxic = "\n\nA non-toxic reply: "
    #print(len(text), type(text))
    if toxic in text:
        return text.split(toxic)[0] + nontoxic
    else:
        return text
        #return text.split(nontoxic)[0] + nontoxic

class CausalLMDataset(Dataset):
    def __init__(self, tokenizer, txt_list, max_length=512, return_text=False):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        self.txt_list = txt_list
        self.return_text = return_text
        if tokenizer:
            for txt in tqdm(txt_list, desc="tokenizing"):
                encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length")
                self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if self.return_text:
            return self.txt_list[idx]
        return self.input_ids[idx], self.attn_masks[idx]


    @classmethod
    def create(cls, dataset, tokenizer, max_length=512):
        enrichment_test = dataset['test'].map(lambda x: {'text': make_non_toxic(x['text'])})
        print("done enrichment")
        return {
            "train": cls(tokenizer, dataset['train']["text"], max_length),
            "test": cls(tokenizer, dataset['test']["text"], max_length),
            "enrichment_test": cls(tokenizer, enrichment_test["text"], max_length),
            "validation": cls(tokenizer, dataset['validation']["text"], max_length)
        }



# fine-tune neogpt model on the dataset
def fine_tune_neogpt(model, tokenizer, dataset, config, log_model="false"):
    # set an environment variable in python
    import os
    os.environ["WANDB_LOG_MODEL"] = log_model
    
    # fine-tune model
    
    # load TrainingArguments parameters from kwargs
    training_args = ORTTrainingArguments(
        output_dir=config.get('output_dir', "./results"),               # output directory
        num_train_epochs=config.get('epochs',3),                        # total number of training epochs
        per_device_train_batch_size=config.get('batch_size', 16),       # batch size per device during training
        per_device_eval_batch_size=config.get('eval_batch_size', 64),   # batch size for evaluation
        warmup_steps=config.get("warmup_steps", 500),                   # number of warmup steps for learning rate scheduler
        weight_decay=config.get("decay", 0.01),                         # strength of weight decay
        logging_dir=config.get("logging_dir", "./logs"),                # directory for storing logs
        save_strategy=config.get("save_strategy", "epoch"),
        save_total_limit=1,
        report_to="wandb",
        optim="adamw_ort_fused",
        log_level="debug",
        learning_rate=2e-05,
    )

    trainer = ORTTrainer(
        model=model,                                  # the instantiated 🤗 Transformers model to be trained
        args=training_args,                           # training arguments, defined above
        train_dataset=dataset["train"],               # training dataset
        eval_dataset=dataset["validation"],
        data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                    'attention_mask': torch.stack([f[1] for f in data]),
                                    'labels': torch.stack([f[0] for f in data])},
        feature="causal-lm-with-past"
    )

    trainer.train()
    trainer.save_model(config.get("save_as", './my_model'))
    tokenizer.save_pretrained(config.get("save_as", './my_model'))
    
    return model

# load the dataset, fine-tune the model, and save the model, then run the test set
def run_neogpt(config, finetune=True, use_onnx=False):
    # login to wandb and pass it the config object
    wandb.login()
    wandb.init(config=config)

    # load model
    tokenizer, model = load_model(config['model_name'], config.get('tokenizer_name'), use_onnx=use_onnx)

    # load dataset
    dataset = load_from_disk(config["dataset_path"])

    # create a new dataset with the non-toxic version of the test set
    print("creating dataset")
    dataset = CausalLMDataset.create(dataset, tokenizer, max_length=512)

    # fine-tune model
    if finetune:
        print("fine-tuning model")
        model = fine_tune_neogpt(model, tokenizer, dataset, config)

if __name__ == "__main__":
    import sys, json
    config_file = "configs/config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    with open(config_file) as f:
        config = json.load(f)
    run_neogpt(config, use_onnx=True)
    #run_neogpt({"model_name": "EleutherAI/gpt-neo-125M", "dataset_path": "data/prochoice.enriched.toxicity"})
    #run_neogpt({"model_name": "./onnx/", #"./results/checkpoint-13500/", 
    #        "tokenizer_name": "./onnx/", 
    #        "dataset_path": "data/prochoice.enriched.toxicity"}, 
    #        finetune=False,
    #        use_onnx=True)
    #run_neogpt({"model_name": "EleutherAI/gpt-neo-125M", "tokenizer_name": "EleutherAI/gpt-neo-125M", "dataset_path": "data/prochoice.data"}, finetune=False)

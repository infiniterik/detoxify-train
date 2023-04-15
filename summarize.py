from transformers import pipeline
import torch
from tqdm import tqdm
import wandb
import pandas as pd

tqdm.pandas()

def acquire_data(path, fname):
    run = wandb.init()
    artifact = run.use_artifact(path, type='dataset')
    artifact_data = artifact.download()
    return pd.read_json(artifact_data+"/"+fname)

def summarize(data, output, model="jordiclive/flan-t5-11b-summarizer-filtered", prompt="Produce a short summary of the following social media post:")
    summarizer = pipeline("summarization", model, torch_dtype=torch.bfloat16, device=0)
    data["summary"] = data["text"].progress_map(summarize_document)
    data.to_json(output)

def summarize_document(raw_document):
    results = summarizer(
        f"{prompt} \n\n {raw_document}",
        num_beams=5,
        min_length=5,
        no_repeat_ngram_size=3,
        truncation=True,
        max_length=min(max(10, len(raw_document)//12), 256),
    )
    #print(raw_document + "\n\t" + results[0]["summary_text"])
    return results[0]["summary_text"]

def run(path, fname, output, model="jordiclive/flan-t5-11b-summarizer-filtered", prompt="Produce a short summary of the following social media post:"):
    data = acquire_data(path, fname)
    print("running summarization:"
    print("\tdata:\t{path}/{fname}".format(path, fname))
    print("\tmodel:\t{model}\n\tprompt:\t{prompt}".format(model, prompt))
    summarize(data, output, model, prompt)
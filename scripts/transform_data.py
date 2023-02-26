import json
import wandb
import click
from tqdm import tqdm

def load_data(filename, is_lines=False):
    with open(filename) as data:
        if is_lines:
            return [json.loads(l) for l in data if l.strip()]
        else:
            return json.load(data)

def detect_removed_post(post):
    """Expects the post to have 'title', 'body', and 'selftext'
    """
    deleted = post.get("title") == "[deleted by user]" 
    self_removed = post.get("selftext") == "[removed]"
    body_removed = post.get("body") == "[removed]"

    return deleted or self_removed or body_removed

def get_text(post):
    title = post.get("title", None)
    body = post.get("body", None)
    selftext = post.get("selftext", None)

    return "".join([x+"\n" for x in [title, body, selftext] if x])

def process_praw(fname, is_lines=False, additional_keys=["id", "parent_id", "author_flair_text", "score"]):
    data = load_data(fname, is_lines)

    result = []
    for d in tqdm(data):
        if not(detect_removed_post(d)):
            point = {}
            for k in additional_keys:
                point[k] = d.get(k)
            point["text"] = get_text(d)
            result.append(point)
    return result

def write(data, output):
    with open("data/"+output, 'w') as out:
        json.dump(data, out)

@click.group()
def cli():
    """Tools to transform praw datasets"""
    pass

@cli.command()
@click.argument("wandb_artifact")
@click.option("-d", "--directory")
def upload(wandb_artifact, directory="data/"):
    """Upload all processed datasets in the output folder to wandb"""
    wandb.init()
    artifact = wandb.Artifact(wandb_artifact, type='dataset')
    artifact.add_dir(directory)
    wandb.log_artifact(artifact)

@cli.command()
@click.argument("fname")
@click.argument("output")
@click.option("-a", "--additional_keys", help="Additional keys to keep beyond 'text' (comma-separated list)", type=str, default="id,parent_id,author_flair_text,score")
@click.option("--is-lines", is_flag=True, help="Whether the input file is in jsonl format")
def process(fname, output, *, additional_keys="id,parent_id,author_flair_text,score", is_lines=False):
    """Process the praw dataset and write it to the data/ folder"""
    data = process_praw(fname, is_lines, additional_keys.split(","))
    write(data, output)


if __name__ == "__main__":
    cli()
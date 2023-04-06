from collections import defaultdict
import re
from datasets import Dataset, DatasetDict

from tqdm import tqdm
tqdm.pandas()

class PostIndex:
    def __init__(self, data):
        self.ix = {x.id: x for x in data}
        self.child_of = defaultdict(list)
        for d in data:
            if d.get("parent_id"):
                self.child_of[d.get("parent_id")[:3]].append(d.get("id"))

    def get_parent(self, post):
        id = post.get("parent_id")
        if id:
            return self.ix.get(id[3:])
        return None
    
    def get_siblings(self, post):
        return self.child_of.get(post.get("parent_id")[3:], [])
    
    def get(self, id):
        return self.ix.get(id)

import pandas as pd

def load_df(dataset):
    df = pd.read_json(dataset)    
    df.parent_id = df.parent_id.map(lambda x: x[3:] if x else None)
    return df

# X is the parent, Y is the child
def attach_parents(df):
    return df.merge(df, left_on='id', right_on='parent_id')

def indicator_to_text(confidence, switch=False):
    if confidence > 0.5:
        if switch:
            return "non-toxic"
        return "toxic"
    else:
        if switch:
            return "toxic"
        return "non-toxic"

class SourceTarget:
    def source(self, row):
        return row['text_x']
    
    def target(self, row):
        return row['text_y']
    
    def __call__(self, df):
        df = df.copy()
        df['text'] = df.apply(lambda x: self.source(x) + " " + self.target(x), axis=1)
        #df['target_text'] = df.apply(self.target, axis=1)
        return df[['text']]
    
    def split_dataset(self, df, train=0.8, test=0.1, val=0.1):
        # shuffle the dataframe
        df = df.sample(frac=1).reset_index(drop=True)
        return df[:int(len(df)*train)], df[int(len(df)*train):int(len(df)*(train+test))], df[int(len(df)*(train+test)):]

    def as_dataset(self, df, train=0.8, test=0.1, val=0.1):
        train, test, validation = self.split_dataset(df, train, test, val)
        return DatasetDict({
            'train': Dataset.from_pandas(self.df(train)),
            'test': Dataset.from_pandas(self.df(test)),
            'validation': Dataset.from_pandas(self.df(validation))
        })

class ChildFromParent(SourceTarget):
    def __init__(self, tokenizer=None, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def source(self, row):
        return "post: " + row['text_x'] + "\nreply:"
    
    def target(self, row):
        return row['text_y']

class ChildFromParentWithToxicity(ChildFromParent):
    def source(self, row):
        return "A " + indicator_to_text(row['enrichments_x'].get("toxicity")) + " post: " + row["text_x"] +  "\nA " + indicator_to_text(row['enrichments_y'].get("toxicity")) + " reply: "
    
    def target(self, row):
        return row['text_y']
    
def make_prochoice_enrichment():
    ds = attach_parents(load_df("data/prochoice.enriched.json"))
    cpt = ChildFromParentWithToxicity()
    cpt.as_dataset(ds).save_to_disk("data/prochoice.enriched.toxicity")

def load_as_df(data="data/prochoice.enriched.json"):
    df = attach_parents(load_df(data))
    cpt = ChildFromParentWithToxicity()
    train, test, val = cpt.split_dataset(df)
    print(val.keys())
    train = train.progress_apply(lambda x: {'source_text': cpt.source(x), "target_text": cpt.target(x)}, axis="columns")
    test = test.progress_apply(lambda x: {'source_text': cpt.source(x), "target_text": cpt.target(x)}, axis="columns")
    val = val.progress_apply(lambda x: {'source_text': cpt.source(x), "target_text": cpt.target(x)}, axis="columns")
    return train, test, val

def make_dataset(source, transform, target):
    ds = attach_parents(load_df(source))
    transform.as_dataset(ds).save_to_disk(target)
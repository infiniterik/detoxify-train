"""Code to transform pandas dataframes into simplet5 format"""

import pandas as pd
from tqdm import tqdm, tqdm_pandas
tqdm.pandas()

func_dict = {}
func_arguments = {}
# A simple function decorator which adds the function to a dictionary of functions
def register_function(func):
    """A simple function decorator which adds the function to a dictionary of functions"""
    func_dict[func.__name__] = func
    register_arguments(func)
    return func

def register_arguments(func):
    """A simple function decorator which adds the function's argument names to a dictionary"""
    func_arguments[func.__name__] = func.__code__.co_varnames
    return func

def apply_function(func, arguments):
    """Applies a function to a dictionary of arguments"""
    try:
        f = func_dict[func]
    except KeyError:
        raise KeyError("Function {} not found in function dictionary".format(func))
    try:
        arg_keys = func_arguments[func]
    except KeyError:
        raise KeyError("Arguments for {} not found in function argument dictionary".format(func))
    calling_args = {a: arguments[a] for a in arg_keys if a in arguments}
    return f(**calling_args)


def load_data(path):
    df = pd.read_json(path)
    df["parent_id"] = df["parent_id"].fillna("").map(lambda x: x[3:])
    return df

# X is the parent, Y is the child
def attach_parents(df):
    return df.merge(df, left_on='id', right_on='parent_id')

def indicator_to_text(confidence, switch=False):
    if confidence > 0.5:
        if switch:
            return "high toxicity"
        return "low toxicity"
    else:
        if switch:
            return "low toxicity"
        return "high toxicity"

def get_simplet5_format(df :  pd.DataFrame, source : str, target : str) -> pd.DataFrame:
    """Transforms a dataframe into a simplet5 format dataframe. Resulting columns should be named source_text and target_text."""
    df = df[[source, target]]
    df.columns = ["source_text", "target_text"]
    return df

@register_function
def get_parent_child(df, parent_column="text_x", 
                         child_column="text_y",
                         parent_prefix="A post: ",
                         child_prefix="A reply: "):
    """Transforms a dataframe into a parent child format dataframe. Resulting columns should be named parent and child."""
    df = attach_parents(df)
    df[parent_column] = df[parent_column].map(lambda x: parent_prefix + x + "\n" + child_prefix)
    return get_simplet5_format(df, parent_column, child_column)

@register_function
def get_parent_child_toxicity(df, parent_column='text_x', child_column='text_y', enrichments_column="enrichments", parent_prefix="A {} post: ", child_prefix="A {} reply: "):
    """Transforms a dataframe into a parent child format dataframe. 
    Resulting columns should be named source_text and target_text.
    """
    df = df.copy() #don't overwrite the original dataframe
    df[enrichments_column] = df[enrichments_column].apply(lambda x: indicator_to_text(x.get("toxicity")))
    df = attach_parents(df)
    print(df)
    
    # construct prompt and response
    p = lambda x: parent_prefix.format(x[enrichments_column+"_x"])
    c = lambda x: "\n" + child_prefix.format(x[enrichments_column+"_y"])
    parent = lambda x: p(x) + x[parent_column] + c(x)

    # transform dataframe
    df[parent_column] = df.apply(parent, axis=1, result_type="reduce")
    # drop and rename
    return get_simplet5_format(df, parent_column, child_column)

@register_function
def get_parent_child_summary(df, parent_column="text_x",
                          child_column="text_y",
                          summary_column="summary",
                          parent_prefix="Post summary: {} post: ",
                          child_prefix="Post reply: {} reply:"):
    """Transforms a dataframe into a parent child format dataframe. 
    Resulting columns should be named source_text and target_text.
    """
    df = attach_parents(df)
    
    # construct prompt and response
    p = lambda x: parent_prefix.format(x[summary_column+"_x"])
    c = lambda x: "\n" + child_prefix.format(x[summary_column+"_y"])
    parent = lambda x: p(x) + "\n" + x[parent_column] + c(x)

    # transform dataframe
    df[parent_column] = df.progress_apply(parent, axis=1, result_type="reduce")
    # drop and rename
    return get_simplet5_format(df, parent_column, child_column)

@register_function
def get_parent_child_toxic_summary(df, parent_column="text_x",
                          child_column="text_y",
                          summary_column="summary",
                          enrichments_column="enrichments",
                          parent_prefix="Post summary: {}\n A {} post: ",
                          child_prefix="Post reply: {}\n A {} reply:"):
    """Transforms a dataframe into a parent child format dataframe. 
    Resulting columns should be named source_text and target_text.
    """
    df = df.copy()
    df[enrichments_column] = df[enrichments_column].apply(lambda x: indicator_to_text(x.get("toxicity")))
    df = attach_parents(df)
    
    # construct prompt and response
    # Summary column is incorrect
    p = lambda x: parent_prefix.format(x[summary_column+"_x"], x[enrichments_column+"_x"])
    c = lambda x: "\n" + child_prefix.format(x[summary_column+"_y"], x[enrichments_column+"_y"])
    parent = lambda x: p(x) + "\n" + x[parent_column] + c(x)

    # transform dataframe
    df[parent_column] = df.progress_apply(parent, axis=1, result_type="reduce")
    # drop and rename
    return get_simplet5_format(df, parent_column, child_column)

def process_data(config):
    data = load_data(config["path"])
    fn = config.get("preprocess", "get_parent_child")
    args = {k: v for k, v in config.get("args", {}).items()}
    args["df"] = data
    return apply_function(fn, args)
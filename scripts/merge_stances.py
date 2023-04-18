import pandas as pd

def load_stance(path):
    df = pd.read_json(path, lines=1)
    df['id'] = df['attribution'].map(lambda x: x['document_id'])
    df = df[['id', 'belief_string', 'sentiment_string', 'belief_type', 'positive_sentiment']]
    return df

def merge(stance, enriched, output_path):
    stance = load_stance(stance)
    enriched = pd.read_json(enriched)
    stance.merge(enriched, on='id').to_json(output_path)

import pandas as pd
import langid
from detoxify import Detoxify
import numpy as np
from transformers import pipeline
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
from anonymization import  remove_url, anonThread
from tqdm import tqdm


def batch_enrich(df, use_gpu=True, chunk_size=32):
    text_df = df
    text_df['proc_text'] = text_df['text'].apply(remove_url)
    #model = BERTopic(min_topic_size=25, calculate_probabilities=False, verbose=True, vectorizer_model=TfidfVectorizer(min_df=5, max_df=0.5, stop_words='english'))
    #topics, probs = model.fit_transform(text_df['proc_text'].tolist())
    #def get_topic_words(top):
    #    if top == -1:
    #        return top
    #    return str(top) + ': ' + ', '.join([single[0] for single in model.get_topic(top)])
    # this saved model can be loaded to reuse the topic model
    #model.save('bertopic_model')
    #text_df['topic'] = topics
    #top_terms = text_df['topic'].apply(get_topic_words).tolist()
    #text_df['topic'] = top_terms
    #print(text_df['topic'].value_counts())

    text_df['lang'] = text_df['proc_text'].apply(lambda x: langid.classify(x)[0])
    detox = Detoxify('multilingual', device='cuda' if use_gpu else 'cpu')
    print('applying detox')
    dt_results = []
    chunks = np.array_split(text_df, int(len(text_df) / chunk_size))

    # device=0 for GPU
    emo_classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True, device=0 if use_gpu else -1)
    #bert_multi_sentiment = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment", return_all_scores=True, device=0 if use_gpu else -1)
    for chunk in tqdm(chunks):
        cur_results = detox.predict(chunk['proc_text'].tolist())
        cur_results = [dict(zip(cur_results,t)) for t in zip(*cur_results.values())]
        for classifier in [emo_classifier]: #[emo_classifier, bert_multi_sentiment]:
            predicted = classifier(chunk['proc_text'].tolist(), batch_size=8, truncation=True)
            for j, sub_l in enumerate(predicted):
                for single in sub_l:
                    cur_results[j][single['label']] = single['score']
        dt_results.extend(cur_results)

    text_df['enrichments'] = dt_results 
    return text_df

def get_text(row):
    text = ''
    if pd.notnull(row['title']):
        text += row['title'] 
        if pd.notnull(row['selftext']):
            text += '\n' + row['selftext']
    else:
        text += row['body']
    return text

def anon_enrich(df, use_gpu=True, chunk_size=32):
    df['text'] = df.apply(get_text, axis=1)
    df = df[df['text'] != '']
    df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    anon = pd.DataFrame(df.apply(lambda row: anonThread(row)[1], axis=1).tolist())
    enr_df = batch_enrich(anon, use_gpu, chunk_size)
    return enr_df


#df = next(pd.read_json('/Users/mackblackburn/PycharmProjects/civil_sanctuary/misc/data/controversial_subs_last_10k.json', lines=True, chunksize=100))
#print(anon_enrich(df))

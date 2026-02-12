import pandas as pd
import os
import re
import nltk
import gensim
import spacy
from gensim.utils import simple_preprocess
from gensim import corpora
from nltk.corpus import stopwords
import pyLDAvis.gensim
import pickle
from collections import Counter

def main():

    # ==========================
    # Load data
    # ==========================
    df = pd.read_csv("/Users/simonkurzewski/Desktop/Statistik-kand/structured_articles_lda.csv")
    df = df.drop(columns=["PublicationType", "Link"], errors="ignore")

    # ==========================
    # Clean text
    # ==========================
    df['Text_processed'] = (
        df['Text']
            .fillna("")
            .apply(lambda x: re.sub('[,\.!?]', '', x))
            .str.lower()
    )

    # ==========================
    # Stopwords
    # ==========================
    nltk.download("stopwords")
    swedish_stop = stopwords.words("swedish")

    swedish_stop.extend([
        "infor","nasta","borjat","gor","for","redo","di","kvar","trots",
        "gar","pa","haller","nar","sitt","an","tva","vill","ga","nytt",
        "medan","dag","valt","ar","enig","galler","isar","ska","fran",
        "mer","kommer","sa","ocksa","andra","finns","aven","dar","manga",
        "over","ser","in","maste","sager","nya","bara","nagot","fram",
        "redan","enligt","gora","senaste","storre","ta","tidigare", "procent",
        "vanta"
    ])

    # ==========================
    # Tokenization
    # ==========================
    def sent_to_words(sentences):
        for sentence in sentences:
            yield simple_preprocess(str(sentence), deacc=True)

    data = df['Text_processed'].tolist()
    data_words = list(sent_to_words(data))

    # Remove stopwords
    data_words = [[w for w in doc if w not in swedish_stop] for doc in data_words]

    # ==========================
    # Lemmatization
    # ==========================
    nlp = spacy.load("sv_core_news_lg", disable=["ner","parser"])

    def lemmatize(docs):
        out = []
        for words in docs:
            doc = nlp(" ".join(words))
            out.append([token.lemma_ for token in doc if token.lemma_ not in swedish_stop])
        return out

    data_words = lemmatize(data_words)

    # ==========================
    # LDA prep
    # ==========================
    id2word = corpora.Dictionary(data_words)
    corpus = [id2word.doc2bow(text) for text in data_words]

    # ==========================
    # LDA model
    # ==========================
    num_topics = 5

    lda_model = gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        iterations=100,
        chunksize=100,
        workers=1   # <-- IMPORTANT on macOS
    )

    print(lda_model.print_topics())

    # ==========================
    # Visualization
    # ==========================
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

    os.makedirs("results", exist_ok=True)
    pyLDAvis.save_html(LDAvis_prepared, f"./results/ldavis_{num_topics}.html")

    # ==========================
    # Topic-word table
    # ==========================

    num_words = 15  # number of top words per topic

    topic_data = []

    for topic_id in range(num_topics):
        words_probs = lda_model.show_topic(topic_id, topn=num_words)
        for word, prob in words_probs:
            topic_data.append({
                "Topic": f"Topic {topic_id + 1}",
                "Word": word,
                "Probability": prob,
                "Percentage": prob * 100
            })
    topic_df = pd.DataFrame(topic_data)
    # Flatten corpus words
    all_words = [w for doc in data_words for w in doc]
    word_counts = Counter(all_words)
    total_words = sum(word_counts.values())

    topic_df["Corpus_Count"] = topic_df["Word"].map(word_counts)

    topic_df["Corpus_Percentage"] = (
    topic_df["Corpus_Count"] / total_words * 100
    )
    topic_df = topic_df.round({
    "Percentage": 2,
    "Corpus_Percentage": 3
    })

    print(topic_df.to_string(index=False))

if __name__ == "__main__":
    main() 



# ================================
# Load Libraries
# ================================
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer, util

print("Libraries loaded script is staring!")

# ================================
# Load data
# ================================
df = pd.read_csv("/Users/simonkurzewski/Desktop/Statistik-kand/structured_articles.csv", usecols=["Text", "Date"])
df["Date"] = pd.to_datetime(df["Date"], format="mixed")

# ADD ArticleID so that i can merge data later
df["ArticleID"] = df.index

print("Data loaded starting semantic sentence analysis")

# ================================
# Semantic similarity
# ================================

# model import
model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")

# Source text (Lists of all sentences)
source = df["Text"].tolist()

# Prompts
positive_prompts = [
    "Regeringen kommer att öka budgeten.",
    "Staten höjer de offentliga utgifterna.",
    "Finanspolitiken blir mer expansiv.",
    "Budgetutrymmet ökar.",
    "Regeringen satsar mer pengar."
]

negative_prompts = [
    "Regeringen minskar budgeten.",
    "Staten ska spara mer.",
    "Finanspolitiken stramas åt.",
    "Offentliga utgifterna minskar.",
    "Nedskärningar i budgeten."
]


# Embeded prompts
emb_source = model.encode(source, batch_size = 32, convert_to_tensor=True, show_progress_bar=True)
emb_pos = model.encode(positive_prompts, batch_size = 32,  convert_to_tensor=True, show_progress_bar=True)
emb_neg = model.encode(negative_prompts, batch_size = 32, convert_to_tensor=True, show_progress_bar=True)


## Compute similarity
# Comopute the cosine similarity
sim_pos = util.cos_sim(emb_source, emb_pos)
sim_neg = util.cos_sim(emb_source, emb_neg)


# Pick maximum sim
df['similarity_positive'] = sim_pos.max(dim=1).values.cpu().numpy()
df['similarity_negative'] = sim_neg.max(dim=1).values.cpu().numpy()


# ================================
# Fiscal relevance + polarity = Article Sentiment 
# ================================
df["relevance"] = df[["similarity_positive", "similarity_negative"]].max(axis=1)
df["polarity"]  = df["similarity_positive"] - df["similarity_negative"]

# Keep only fiscal-related sentences
df["ArticleSentiment"] = df["polarity"]*df["relevance"]


# ================================
# Monthly Fiscal Sentiment Index
# ================================
df["Month"] = df["Date"].dt.to_period("M")

FSI = df.groupby("Month")["ArticleSentiment"].mean().reset_index()
FSI.rename(columns={"ArticleSentiment": "FiscalSentimentIndex"}, inplace=True)


# ================================
# Save files
# ================================
df.to_csv("articles_with_sentiment.csv", index=False)
FSI.to_csv("fiscal_sentiment_index.csv", index=False)

print("🎉 All done!")
print("- articles_with_sentiment.csv")
print("- fiscal_sentiment_index.csv")


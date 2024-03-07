# Import of library named spacy
import spacy
# Import of library named pandas, wi will be calling it with shortcut pd
import pandas as pd
# of TextBlob from textblob
from textblob import TextBlob


nlp = spacy.load('en_core_web_md')


def sentence_similarity(sentence1, sentence2):
    # This def calculates similarity between 2 sentences
    return nlp(sentence2).similarity(nlp(sentence1))


def analyze_polarity(text):
    # We preprocess text, switch everything to lower case and remove unnecessary characters at edges of string
    doc = nlp(text.lower().strip())
    # now we remove stop words and punctuation
    processed = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    # and we get simplified sentence
    processed = " ".join(processed)
    
    # Analyze polarity with TextBlob
    print(TextBlob(text).sentiment.polarity)  # <=== Un-comment for polarity of base text
    polarity = TextBlob(processed).sentiment.polarity

    """
    Returns sentiment based on calculated polarity 
    
    Capstone Project Assignment: A polarity score of 1 indicates a very positive sentiment, while a polarity score of -1 indicates a very negative sentiment.
    A polarity score of 0 indicates a neutral sentiment.

    This is not right. It would make neutral sentiment unachievable. I had to include bias, but still got this result:
    "I returned it the next day"  >>>  positive sentiment
    """
    if polarity > 0.41:
        sentiment = 'positive'
    elif polarity < 0.1:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return f"Text: {text}\nPolarity score: {polarity}\nSentiment: {sentiment}"


# First we read our file
df = pd.read_csv('1429_1.csv')

# We only need 'reviews.text', but I decided to include other two to help us determine accuracy of our model
df = df[['reviews.text', 'reviews.doRecommend', 'reviews.rating']]

# clean-up, NaN's removed
df = df.dropna(subset=['reviews.text'])

# We are not going to analyse whole thing. Next line picks some reviews from dataframe
for i in [1, 2, 3, 6, 222, 226, 117, 281, 126, 169]:
    # prints current row
    print(df.iloc[i], '\n')
    # prints result of polarity / sentiment analysis
    print(analyze_polarity(df.iloc[i, 0]), '\n------------------\n')

# 2 sentences for similarity analysis
review1 = df.iloc[1, 0]
review2 = df.iloc[2, 0]
# Prints result
print('Review1:', review1, '\nReview2:', review2, '\nSimilarity between Review1 and Review2:', sentence_similarity(review1, review2))












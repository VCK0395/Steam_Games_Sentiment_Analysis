import pandas as pd
import numpy as np
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

steam_r = pd.read_csv('Steam_Games_Reviews.csv')

steam_r_new = steam_r[['recommendation', 'review', 'title']]

def Subjective(text):
        return TextBlob(text).sentiment.subjectivity


def Polarity(text):
    return TextBlob(text).sentiment.polarity



steam_r_new['Subjectivity'] = steam_r_new['review'].apply(Subjective)

steam_r_new['Polarity'] = steam_r_new['review'].apply(Polarity)

print(steam_r_new)

allwords = ' '.join([texts for texts in steam_r_new['review']])

my_cloud = WordCloud(background_color='black', random_state= 21).generate(allwords)

plt.imshow(my_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

def Analysis(numbers):
    if numbers > 0:
        return 'Positive'
    elif numbers == 0:
        return 'Neutral'
    else:
       return 'Negative'

steam_r_new['Analysis'] = steam_r_new['Polarity'].apply(Analysis)

print(steam_r_new[['Polarity', 'Analysis']])

vect = TfidfVectorizer(max_features=200, stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2), token_pattern=r'\b[^\d\W][^\d\W]+\b')
vect.fit(steam_r_new['review'])
X_review = vect.transform(steam_r_new['review'])
review_transformed =pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())

review_transformed['recommendation_score'] = steam_r_new['recommendation'].apply(lambda val: 0 if val == 'Not Recommended' else 1)

X = review_transformed.drop('recommendation_score', axis=1)
y = review_transformed['recommendation_score']


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=456)

# Train a logistic regression
log_reg = LogisticRegression().fit(X_train, y_train)
# Predict the labels
y_predicted = log_reg.predict(X_test)

# Print accuracy score and confusion matrix on test set
print('Accuracy on the test set: ', accuracy_score(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted)/len(y_test))

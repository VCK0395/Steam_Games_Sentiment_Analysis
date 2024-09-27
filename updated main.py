# Import Packages
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Import CSV
steam_r = pd.read_csv('Steam_Games_Reviews.csv')

steam_r_new = steam_r[['recommendation', 'review', 'title']]

# Optimized sentiment analysis using VADER
analyzer = SentimentIntensityAnalyzer()

def vader_polarity(text):
    return analyzer.polarity_scores(text)['compound']

steam_r_new['Polarity'] = steam_r_new['review'].apply(vader_polarity)

# Word Cloud generation
allwords = ' '.join(steam_r_new['review'])
my_cloud = WordCloud(background_color='black', random_state=12).generate(allwords)

plt.imshow(my_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Analysis Function
def sentiment_label(score):
    if score > 0:
        return 'Positive'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Negative'

steam_r_new['Analysis'] = steam_r_new['Polarity'].apply(sentiment_label)

# Vectorization with optimizations
vect = TfidfVectorizer(max_features=150, stop_words='english', ngram_range=(1, 2), token_pattern=r'\b\w+\b')
X_review = vect.fit_transform(steam_r_new['review'])

# Encoding recommendation to binary
y = steam_r_new['recommendation'].apply(lambda val: 0 if val == 'Not Recommended' else 1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_review, y, test_size=0.2, random_state=456)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_predicted = log_reg.predict(X_test)

# Print accuracy score and confusion matrix
print('Accuracy on the test set: ', accuracy_score(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted)/len(y_test))

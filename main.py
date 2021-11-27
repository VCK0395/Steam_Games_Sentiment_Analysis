# Import Packages
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Import CVS
steam_r = pd.read_csv('Steam_Games_Reviews.csv')

steam_r_new = steam_r[['recommendation', 'review', 'title']]

# Create Subjectivity and Polarity Function
def Subjective(text):
        return TextBlob(text).sentiment.subjectivity


def Polarity(text):
    return TextBlob(text).sentiment.polarity

steam_r_new['Subjectivity'] = steam_r_new['review'].apply(Subjective)

steam_r_new['Polarity'] = steam_r_new['review'].apply(Polarity)

print(steam_r_new)

#Create and plot the Word Cloud
allwords = ' '.join([texts for texts in steam_r_new['review']])

my_cloud = WordCloud(background_color='black', random_state=12).generate(allwords)

plt.imshow(my_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Create A positive, neutral, and negative analysis function
def Analysis(numbers):
    if numbers > 0:
        return 'Positive'
    elif numbers == 0:
        return 'Neutral'
    else:
       return 'Negative'

steam_r_new['Analysis'] = steam_r_new['Polarity'].apply(Analysis)

print(steam_r_new[['Polarity', 'Analysis']])

# Create a bag of words and put the words in a DataFrame
vect = TfidfVectorizer(max_features=200, stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2), token_pattern=r'\b[^\d\W][^\d\W]+\b')
vect.fit(steam_r_new['review'])
X_review = vect.transform(steam_r_new['review'])
review_transformed = pd.DataFrame(X_review.toarray(), columns=vect.get_feature_names())

# Create encoding binary variable function on recommendation
review_transformed['recommendation_score'] = steam_r_new['recommendation'].apply(lambda val: 0 if val == 'Not Recommended' else 1)
print(review_transformed)

# Set up the Train/test split and logistic regression model
X = review_transformed.drop('recommendation_score', axis=1)
y = review_transformed['recommendation_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=456)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_predicted = log_reg.predict(X_test)

# Print accuracy score and confusion matrix on test set
print('Accuracy on the test set: ', accuracy_score(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted)/len(y_test))

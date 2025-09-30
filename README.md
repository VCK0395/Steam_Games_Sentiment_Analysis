# 🎮 Steam Game Review Analyzer and Predictor

This project uses Natural Language Processing (NLP) techniques, including Sentiment Analysis with **TextBlob** and a **Logistic Regression** classifier with **TF-IDF**, to analyze Steam game reviews. It extracts subjective and polar sentiment and predicts whether a user will recommend a game based on their review text.

---

## 💡 What It Does (The What)

This script performs two main functions on a dataset of Steam game reviews (`Steam_Games_Reviews.csv`):

1.  **Sentiment Analysis & Visualization:**
    * Calculates **Subjectivity** (how opinionated the text is) and **Polarity** (how positive or negative the sentiment is) using the **TextBlob** library.
    * Categorizes reviews into 'Positive', 'Neutral', or 'Negative' based on Polarity.
    * Generates and plots a **Word Cloud** to visually represent the most frequent words in the entire corpus of reviews.

2.  **Recommendation Prediction (Text Classification):**
    * Uses **TF-IDF** (**Term Frequency-Inverse Document Frequency**) to convert the text reviews into numerical feature vectors (a **Bag-of-Words** model).
    * Trains a **Logistic Regression** model to predict the original user **recommendation** (`Recommended` vs. `Not Recommended`) using the transformed review text as features.
    * Evaluates the model's performance using **Accuracy Score** and a **Confusion Matrix**.

---

## 🛠️ How to Use It (The How)

### Prerequisites

You need a Python environment with the following libraries installed:

* **Pandas**
* **TextBlob**
* **WordCloud**
* **Matplotlib**
* **Scikit-learn**

### Setup and Installation

1.  **Clone the Repository** (or save the code as a Python file, e.g., `steam_analyzer.py`):
    ```bash
    git clone [your-repo-url]
    cd [your-repo-name]
    ```

2.  **Ensure Dataset is Present:**
    The script requires the dataset file named **`Steam_Games_Reviews.csv`** to be in the same directory as the script.

3.  **Install Dependencies:**
    ```bash
    pip install pandas textblob wordcloud matplotlib scikit-learn
    # Note: TextBlob often requires its corpus data:
    python -m textblob.download_corpora
    ```

### Running the Script

Execute the Python file from your terminal:

```bash
python main.py
```
## 🎯 Why This Project 

Practical NLP Application
This project showcases fundamental and powerful techniques used in Natural Language Processing and Machine Learning:

1.  Understand User Intent: Sentiment analysis is crucial for quickly gauging public opinion or user satisfaction. It transforms unstructured text into measurable, quantifiable data (Polarity and Subjectivity).

2.  Feature Engineering from Text: The use of TF-IDF demonstrates the essential step of turning raw, qualitative text data into meaningful numerical features that an ML model can effectively learn from.

3.  Building a Predictive Tool: The final Logistic Regression model validates that the text content of a review is a strong predictor of the user's explicit recommendation, which is valuable for game developers, marketers, and platform analysts.

By combining visualization (Word Cloud), data labeling (Sentiment Analysis), and predictive modeling (Logistic Regression), the project provides a comprehensive solution for analyzing and acting upon large volumes of user feedback.

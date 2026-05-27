# ============================================================
# CAREER INTELLIGENT SYSTEM
# RESUME BASED CAREER PREDICTION USING ML + NLP
# ============================================================

# INSTALL REQUIRED LIBRARIES
# !pip install pandas numpy scikit-learn nltk

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import re
import nltk
import warnings

warnings.filterwarnings('ignore')

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# DOWNLOAD STOPWORDS
# ============================================================

nltk.download('stopwords')

# ============================================================
# LOAD ONLINE DATASET
# ============================================================

print("\nLoading Online Resume Dataset...\n")

# Online CSV dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json"

# Load dataset
df = pd.read_json(url)

print("\n================ DATASET LOADED ================\n")

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns)

# ============================================================
# CREATE RESUME DATASET FORMAT
# ============================================================

# Use content as resume
df['Resume'] = df['content']

# Use target as career category
df['Category'] = df['target_names']

# Keep only required columns
df = df[['Resume', 'Category']]

# Remove null values
df.dropna(inplace=True)

print("\nProcessed Dataset Shape:")
print(df.shape)

# ============================================================
# TEXT CLEANING FUNCTION
# ============================================================

stop_words = set(stopwords.words('english'))

def clean_resume(text):

    text = str(text)

    # Remove URLs
    text = re.sub(r'http\S+', ' ', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove numbers
    text = re.sub(r'\d+', ' ', text)

    # Convert lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove stopwords
    words = text.split()

    cleaned_words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(cleaned_words)

# ============================================================
# CLEAN RESUME TEXT
# ============================================================

print("\nCleaning Text...\n")

df['Cleaned_Resume'] = df['Resume'].apply(clean_resume)

print("Cleaning Completed!")

# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

df['Category_Encoded'] = label_encoder.fit_transform(df['Category'])

# ============================================================
# FEATURES AND LABELS
# ============================================================

X = df['Cleaned_Resume']
y = df['Category_Encoded']

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================================
# MACHINE LEARNING PIPELINE
# ============================================================

model = Pipeline([
    (
        'tfidf',
        TfidfVectorizer(max_features=5000)
    ),
    (
        'classifier',
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    )
])

# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Machine Learning Model...\n")

model.fit(X_train, y_train)

print("Model Training Completed!")

# ============================================================
# MODEL EVALUATION
# ============================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n================ MODEL ACCURACY ================\n")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")

print(classification_report(y_test, predictions))

# ============================================================
# CAREER PREDICTION FUNCTION
# ============================================================

def predict_career(resume_text):

    cleaned_text = clean_resume(resume_text)

    prediction = model.predict([cleaned_text])

    prediction_id = int(prediction[0])

    predicted_category = label_encoder.inverse_transform([prediction_id])

    return predicted_category[0]

# ============================================================
# SAMPLE TEST
# ============================================================

sample_resume = """
Experienced Python Developer skilled in
Machine Learning, Deep Learning,
TensorFlow, NLP, SQL and Flask.
"""

result = predict_career(sample_resume)

print("\n================ SAMPLE PREDICTION ================\n")

print("Predicted Career Domain:")
print(result)

# ============================================================
# USER INPUT
# ============================================================

print("\n===================================================")
print("ENTER YOUR RESUME TEXT")
print("===================================================\n")

user_resume = input()

output = predict_career(user_resume)

print("\n===================================================")
print("PREDICTED CAREER DOMAIN")
print("===================================================\n")

print(output)

# ============================================================
# AVAILABLE DOMAINS
# ============================================================

print("\n===================================================")
print("AVAILABLE DOMAINS")
print("===================================================\n")

print(df['Category'].unique())

# ============================================================
# END OF PROJECT
# ============================================================
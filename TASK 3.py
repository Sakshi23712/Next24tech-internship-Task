# Install NLTK library if you haven't already
# !pip install nltk

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

# Example dataset (replace with your own dataset)
texts = ["This movie is great!", "I did not like this product.", "Neutral tweet about something."]

# Labels for the examples (0: negative, 1: neutral, 2: positive)
labels = [2, 0, 1]

# Text preprocessing and feature extraction
def preprocess(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    return ' '.join(tokens)

# Preprocess all texts
processed_texts = [preprocess(text) for text in texts]

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_texts)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Initialize Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

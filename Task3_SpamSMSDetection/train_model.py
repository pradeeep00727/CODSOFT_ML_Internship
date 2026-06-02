import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only needed columns
data = data[['v1', 'v2']]

# Rename columns
data.columns = ['label', 'message']

# Convert labels
data['label'] = data['label'].map({
    'ham': 0,
    'spam': 1
})

# Features and labels
X = data['message']
y = data['label']

# Better TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english',
    lowercase=True,
    ngram_range=(1,2)
)

X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Better classifier
model = LinearSVC()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Advanced model saved successfully!")
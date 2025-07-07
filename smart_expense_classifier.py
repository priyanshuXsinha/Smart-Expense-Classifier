# Smart Expense Classifier (Starter Project)

# Step 1: Sample training data (can be expanded)
data = [
    ("Uber ride to airport ₹500", "Travel"),
    ("Pizza from Dominos ₹250", "Food"),
    ("Monthly Netflix subscription ₹499", "Entertainment"),
    ("Bought milk and eggs ₹80", "Groceries"),
    ("Petrol refill ₹1200", "Travel"),
    ("Lunch with friends ₹300", "Food")
]

# Step 2: Train basic model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

texts, labels = zip(*data)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

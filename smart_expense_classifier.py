# Smart Expense Classifier (Starter Project)

# Step 1: Sample training data (can be expanded)
data = [
    # Travel
    ("Uber ride to airport ₹500", "Travel"),
    ("Petrol refill ₹1200", "Travel"),
    ("Flight to Mumbai ₹3500", "Travel"),
    ("Train ticket ₹800", "Travel"),
    ("Taxi from station ₹300", "Travel"),

    # Food
    ("Pizza from Dominos ₹250", "Food"),
    ("Lunch with friends ₹300", "Food"),
    ("Burger from McDonalds ₹150", "Food"),
    ("Dinner at KFC ₹450", "Food"),
    ("Ice cream ₹80", "Food"),

    # Entertainment
    ("Monthly Netflix subscription ₹499", "Entertainment"),
    ("Movie ticket ₹200", "Entertainment"),
    ("Concert pass ₹1500", "Entertainment"),
    ("Prime Video yearly plan ₹999", "Entertainment"),
    ("Cricket match ticket ₹700", "Entertainment"),

    # Groceries
    ("Bought milk and eggs ₹80", "Groceries"),
    ("Vegetables from market ₹120", "Groceries"),
    ("Rice and wheat ₹500", "Groceries"),
    ("Fruits from shop ₹200", "Groceries"),
    ("Sugar and tea ₹150", "Groceries"),
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


import pickle

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Function to predict expense category
def predict_expense(text):
    X_test = vectorizer.transform([text])
    return model.predict(X_test)[0]

# Example test
while True:
    user_input = input("Enter expense description (or 'quit'): ")
    if user_input.lower() == "quit":
        break
    category = predict_expense(user_input)
    print(f"Predicted Category: {category}")

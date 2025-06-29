import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import re

# Step 1: Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Step 2: Load datasets
df_fake = pd.read_csv("C:/Users/Ajul/OneDrive/fakenewsproject/Fake.csv")
df_true = pd.read_csv("C:/Users/Ajul/OneDrive/fakenewsproject/True.csv")

# Step 3: Add labels
df_fake["label"] = 0
df_true["label"] = 1

# Step 4: Combine and shuffle
df = pd.concat([df_fake, df_true])
df = df.sample(frac=1).reset_index(drop=True)

# Step 5: Combine title + text
df["combined"] = df["title"] + " " + df["text"]
df["combined"] = df["combined"].apply(clean_text)

# Step 6: Features and labels
X = df["combined"]
y = df["label"]

# Step 7: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 9: Train model
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_vec, y_train)

# Step 10: Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test_vec))
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")

# Step 11: Prediction loop
while True:
    print("\n🔎 Enter a news article (or type 'exit' to quit):")
    user_input = input()
    
    if user_input.lower() == "exit":
        break
    
    cleaned_input = clean_text(user_input)
    input_vector = vectorizer.transform([cleaned_input])
    prediction = model.predict(input_vector)
    
    if prediction[0] == 1:
        print("✅ The news is REAL.")
    else:
        print("❌ The news is FAKE.")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv("data/maternal_dataset.csv")

X = data.drop("gestational_diabetes", axis=1)
y = data["gestational_diabetes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base models
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
lr = LogisticRegression(max_iter=1000)

# Ensemble model
model = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
    voting='soft'
)

model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("models/diabetes_model.pkl", "wb"))

print("Model trained and saved!")
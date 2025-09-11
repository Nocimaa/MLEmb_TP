from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

texts = [
    "I love this movie!", "This is terrible...", "Amazing performance",
    "Really bad product", "I enjoyed it", "Worst experience ever",
    "Pretty good, I liked it", "Awful, do not watch", "Excellent acting",
    "Not my taste", "Superb story", "Horrible scenes", "Loved the visuals",
    "Disappointing plot", "Fantastic direction", "Bad dialogues",
    "Outstanding soundtrack", "Mediocre acting", "Great cinematography", 
    "Terrible editing"
]

labels = [5, 1, 5, 1, 4, 1, 4, 1, 5, 2, 5, 1, 5, 2, 5, 2, 5, 3, 5, 1]

def train_and_save(save_path="multi_class_model.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    clf = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000
    )
    clf.fit(X_train_vect, y_train)

    preds = clf.predict(X_test_vect)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Accuracy: {acc:.2f}")

    joblib.dump((vectorizer, clf), save_path)
    print(f"Modèle sauvegardé dans {save_path}")

def load_model(save_path="multi_class_model.pkl"):
    vectorizer, clf = joblib.load(save_path)
    return vectorizer, clf

if __name__ == "__main__":
    train_and_save()

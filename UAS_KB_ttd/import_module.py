import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Modul 1: Preprocessing Data
def preprocess_images(image_folder, img_size=(128, 128)):

    data = []
    labels = []
    classes = os.listdir(image_folder)
    for label, class_name in enumerate(classes):
        class_folder = os.path.join(image_folder, class_name)
        for img_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)
            data.append(img.flatten())
            labels.append(label)
    return np.array(data), np.array(labels), classes

# Modul 2: Membagi Data
def split_data(features, labels, test_size=0.3, random_state=42):
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)

# Modul 3: Training Model
def train_random_forest(X_train, y_train, n_estimators=100):

    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

# Modul 4: Evaluasi Model
def evaluate_model(model, X_test, y_test, class_names):

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")

    unique_labels = np.unique(np.concatenate((y_test, y_pred)))
    filtered_class_names = [class_names[i] for i in unique_labels]

    print(classification_report(y_test, y_pred, target_names=filtered_class_names))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=filtered_class_names, yticklabels=filtered_class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    return accuracy

# Modul 5: Deteksi Tanda Tangan
def detect_signature(model, image_path, class_names, img_size=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img_flattened = img.flatten().reshape(1, -1)

    prediction = model.predict(img_flattened)
    predicted_class = class_names[prediction[0]]
    return predicted_class

# Main Program
if __name__ == "__main__":

    image_folder = "TTD_A"

    # Step 1: Preprocess data
    print("Preprocessing images...")
    features, labels, class_names = preprocess_images(image_folder)

    # Step 2: Split data
    print("Splitting data into training and validation sets...")
    X_train, X_test, y_train, y_test = split_data(features, labels)

    # Step 3: Train models
    print("Training models...")
    rf_model = train_random_forest(X_train, y_train)

    # Step 4: Evaluate models
    print("Evaluating Random Forest model...")
    rf_accuracy = evaluate_model(rf_model, X_test, y_test, class_names)

    # Step 5: Detect signature
    print("Detecting signature...")
    test_image_path = input("vell.jpg")
    if os.path.exists(test_image_path):
        owner = detect_signature(rf_model, test_image_path, class_names)
        print(f"The signature belongs to: {owner}")
    else:
        print("The provided image path does not exist.")
# Import modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Preprocessing Data
# Load images and split into training and testing sets
def read_data():
    try:
        # Gunakan engine sesuai dengan format file
        df = pd.read_excel('Kelulusan_Test.XLS', engine='xlrd')  # Jika file .xls
        # df = pd.read_excel('Kelulusan_Test.xlsx', engine='openpyxl')  # Jika file .xlsx
        return df
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return None

def encode_data(df):
    try:
        # Encode non-numeric columns
        label_encoders = {}
        for column in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
        return df, label_encoders
    except Exception as e:
        print(f"Error encoding data: {e}")
        return None, None

def prepare_data():
    df = read_data()
    if df is not None:
        try:
            df, _ = encode_data(df)  # Encode categorical data
            X = df.iloc[:, 2:-1].values  # Features (columns IPS1 to IPK)
            y = df.iloc[:, -1].values  # Labels (last column: STATUS KELULUSAN)
            return X, y
        except Exception as e:
            print(f"Error preparing data: {e}")
            return None, None
    return None, None

# Build Classification Algorithm
def knn(X_train, y_train, X_test):
    try:
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        return knn.predict(X_test)
    except Exception as e:
        print(f"Error in KNN model: {e}")
        return None

def svm(X_train, y_train, X_test):
    try:
        svm = SVC(kernel='linear')
        svm.fit(X_train, y_train)
        return svm.predict(X_test)
    except Exception as e:
        print(f"Error in SVM model: {e}")
        return None

def random_forest(X_train, y_train, X_test):
    try:
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train, y_train)
        return rf.predict(X_test)
    except Exception as e:
        print(f"Error in Random Forest model: {e}")
        return None

# Function to display confusion matrix
def display_confusion_matrix(y_test, y_pred, title):
    try:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(title)
        plt.show()
    except Exception as e:
        print(f"Error displaying confusion matrix: {e}")

if __name__ == '__main__':
    # Load data
    X, y = prepare_data()

    if X is not None and y is not None:
        try:
            # Split dataset into training and testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Classification with KNN
            y_pred_knn = knn(X_train, y_train, X_test)
            if y_pred_knn is not None:
                print("KNN Classification Report:")
                print(classification_report(y_test, y_pred_knn))
                display_confusion_matrix(y_test, y_pred_knn, "KNN Confusion Matrix")

            # Classification with SVM
            y_pred_svm = svm(X_train, y_train, X_test)
            if y_pred_svm is not None:
                print("SVM Classification Report:")
                print(classification_report(y_test, y_pred_svm))
                display_confusion_matrix(y_test, y_pred_svm, "SVM Confusion Matrix")

            # Classification with Random Forest
            y_pred_rf = random_forest(X_train, y_train, X_test)
            if y_pred_rf is not None:
                print("Random Forest Classification Report:")
                print(classification_report(y_test, y_pred_rf))
                display_confusion_matrix(y_test, y_pred_rf, "Random Forest Confusion Matrix")

        except Exception as e:
            print(f"Error in the main execution: {e}")
    else:
        print("Data preparation failed. Please check the input file and format.")

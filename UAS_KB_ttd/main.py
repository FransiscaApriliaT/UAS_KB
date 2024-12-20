from preprocess_images import preprocess_images
from split_data import split_data
from train_random_forest import train_random_forest
from evaluate_model import evaluate_model
from detect_signature import detect_signature
import os

if __name__ == "__main__":
    import os

    image_folder = "ttd"


    print("Preprocessing images...")
    from preprocess_images import preprocess_images
    features, labels, class_names = preprocess_images(image_folder)

 
    print("Splitting data into training and validation sets...")
    from split_data import split_data
    X_train, X_test, y_train, y_test = split_data(features, labels)


    print("Training models...")
    from train_random_forest import train_random_forest
    rf_model = train_random_forest(X_train, y_train)


    print("Evaluating Random Forest model...")
    from evaluate_model import evaluate_model
    rf_accuracy = evaluate_model(rf_model, X_test, y_test, class_names)


    print("Detecting signature...")
    from detect_signature import detect_signature
    test_image_path = input("Enter the path to the signature image: ")
    if os.path.exists(test_image_path):
        owner = detect_signature(rf_model, test_image_path, class_names)
        print(f"The signature belongs to: {owner}")
    else:
        print("The provided image path does not exist.")

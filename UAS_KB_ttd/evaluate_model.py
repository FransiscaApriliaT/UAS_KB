def evaluate_model(model, X_test, y_test, class_names):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

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

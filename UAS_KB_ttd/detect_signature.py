def detect_signature(model, image_path, class_names, img_size=(128, 128)):
    import cv2

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img_flattened = img.flatten().reshape(1, -1)

    prediction = model.predict(img_flattened)
    predicted_class = class_names[prediction[0]]
    return predicted_class
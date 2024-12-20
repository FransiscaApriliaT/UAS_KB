def preprocess_images(image_folder, img_size=(128, 128)):
    import os
    import cv2
    import numpy as np

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
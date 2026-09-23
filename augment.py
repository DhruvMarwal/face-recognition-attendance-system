import os
import cv2
import numpy as np

def augment_image(img):
    augmentations = []
    augmentations.append(img)
    flipped = cv2.flip(img, 1)
    augmentations.append(flipped)
    kernel_sharp = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel_sharp)
    augmentations.append(sharpened)
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    augmentations.append(gaussian)
    median = cv2.medianBlur(img, 5)
    augmentations.append(median)
    kernel_highpass = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])
    highpass = cv2.filter2D(img, -1, kernel_highpass)
    augmentations.append(highpass)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=1)
    augmentations.append(dilated)
    eroded = cv2.erode(img, kernel, iterations=1)
    augmentations.append(eroded)
    return augmentations

def augment_and_save_images(input_folder="faces", output_folder="augmented_faces", image_size=(300, 300)):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    for person in sorted(os.listdir(input_folder)):
        person_path = os.path.join(input_folder, person)
        if not os.path.isdir(person_path):
            continue

        save_path = os.path.join(output_folder, person)
        os.makedirs(save_path, exist_ok=True)

        img_count = 0
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                face_img = img[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, image_size)
                augmented_images = augment_image(face_img)

                for i, aug in enumerate(augmented_images):
                    aug_filename = f"{person}_aug{img_count}_{i}.jpg"
                    cv2.imwrite(os.path.join(save_path, aug_filename), aug)
                img_count += 1

if __name__ == "__main__":
    augment_and_save_images()

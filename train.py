import os
import cv2
import joblib
import numpy as np
from keras_facenet import FaceNet
from sklearn.svm import SVC

embedder = FaceNet()

def get_embedding(img):
    img = cv2.resize(img, (160,160))
    img = np.expand_dims(img, axis=0)
    embedding = embedder.embeddings(img)
    return embedding[0]

dataset_path = "faces"
X, y = [], []
label_map = {}
label_count = 0

for person in sorted(os.listdir(dataset_path)):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue
    label_map[label_count] = person
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        embedding = get_embedding(img)
        X.append(embedding)
        y.append(label_count)
    label_count += 1

X = np.array(X)
y = np.array(y)

clf = SVC(kernel='linear', probability=True)
clf.fit(X, y)

joblib.dump(clf, "face_svm.pkl")
joblib.dump(label_map, "label_map.pkl")
print("✅ Training complete! Models saved.")
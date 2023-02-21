import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR = "C:/Users/Tharu/Documents/ML/datasets/plants"

CATEGORIES = ["plant01", "plant02"]

for category in CATEGORIES:  # do plant1 and plant2
    path = os.path.join(DATADIR,category)  # create path to plant1 and plant2
    for img in os.listdir(path):  # iterate over each image per plant1 and plant2
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)  # convert to array
        plt.imshow(img_array, cmap='gray')  # graph it
        plt.show()  # display!

        break  # we just want one for now so break
    break  #...and one more!
print(img_array)
print(img_array.shape)
IMG_SIZE = 30
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap = "gray")
plt.show()
training_data = []

def create_training_data():
    for category in CATEGORIES:  # do plant1 and plant2

        path = os.path.join(DATADIR,category)  # create path to plant1 and plant2
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=plqant1 1=plant2

        for img in tqdm(os.listdir(path)):  # iterate over each image per plant1 and plant2
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()

print(len(training_data))

import random

random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

p = []
q = []

for features,label in training_data:
    p.append(features)
    q.append(label)

print(p[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

p = np.array(p).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

import pickle

pickle_out = open("p.pickle","wb")
pickle.dump(p, pickle_out)
pickle_out.close()

pickle_out = open("q.pickle","wb")
pickle.dump(q, pickle_out)
pickle_out.close()

pickle_in = open("p.pickle","rb")
p = pickle.load(pickle_in)

pickle_in = open("q.pickle","rb")
q = pickle.load(pickle_in)
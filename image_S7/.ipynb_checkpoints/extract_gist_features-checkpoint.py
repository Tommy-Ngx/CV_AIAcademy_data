import os
import numpy as np
import cv2
import gist
import pickle

DATA = '15-Scene'
cnt = -1

features = []
label = []
for c in os.listdir(DATA):
    cnt += 1
    for file_name in os.listdir(os.path.join(DATA, c)):
        file_path = os.path.join(DATA, c, file_name)
        print(file_name)
        img = cv2.imread(file_path)
        label.append(cnt)
        features.append(gist.extract(img))

print(np.array(features).shape)
print(np.array(label).shape)
pickle.dump(features, open('features.dump', 'wb'))
pickle.dump(label, open('labels.dump', 'wb'))

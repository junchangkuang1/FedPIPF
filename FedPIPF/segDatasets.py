import os
import cv2
import random
filenames = os.listdir(r"G:\second_datasets\LNDb_slice")
filenames_cut = [f.split('_')[0] for f in filenames]
Non_repeat = list(set(filenames_cut))
random.seed(6)
random.shuffle(Non_repeat)

flag = len(Non_repeat) / 10
train = []
val = []
test = []
for i in range(len(Non_repeat)):
    if i < flag * 7:
        train.append(Non_repeat[i])
    elif i > flag * 7 and i <= flag * 8:
        val.append(Non_repeat[i])
    else:
        test.append(Non_repeat[i])

trainsets = []
valsets = []
testsets = []
for i in range(len(filenames_cut)):
    if filenames_cut[i] in train:
        trainsets.append(filenames[i])
    elif filenames_cut[i] in val:
        valsets.append(filenames[i])
    else:
        testsets.append(filenames[i])

for t in trainsets:
    with open(r"G:\second_datasets\LNDb_train.txt",'a') as f1:
        f1.write(t)
        f1.write('\n')
for v in valsets:
    with open(r"G:\second_datasets\LNDb_val.txt",'a') as f2:
        f2.write(v)
for tt in testsets:
    with open(r"G:\second_datasets\LNDb_test.txt",'a') as f3:
        f3.write(tt)
        f3.write('\n')





import os

JM_train = open(r"F:\pythonProject\FedDUS\public_datasettxt_tvt\JM_train.txt",'r').readlines()
JM_train = [t.replace('npz\n','')[:13] for t in JM_train]
JM_train = list(set(JM_train))

JM_val = open(r"F:\pythonProject\FedDUS\public_datasettxt_tvt\JM_valid.txt",'r').readlines()
JM_val = [t.replace('npz\n','')[:13] for t in JM_val]
JM_val = list(set(JM_val))

JM_test = open(r"F:\pythonProject\FedDUS\public_datasettxt_tvt\JM_test.txt",'r').readlines()
JM_test = [t.replace('npz\n','')[:13] for t in JM_test]
JM_test = list(set(JM_test))
print(JM_test)
JM_train_jpg = os.listdir(r"G:\second_datasets\jpg\JM\images")

train_l = []
val_l = []
test_l = []
for j in JM_train_jpg:
    j_name = j[0:13]
    print(j_name)
    if j_name in JM_train:
        train_l.append(j)
    elif j_name in JM_val:
        val_l.append(j)
    else:
        test_l.append(j)
print(len(train_l)+len(val_l)+len(test_l))
for t in train_l:
    with open(r"G:\second_datasets\jpg\JM_train.txt",'a') as f1:
        f1.write(t)
        f1.write('\n')
for v in val_l:
    with open(r"G:\second_datasets\jpg\JM_val.txt",'a') as f2:
        f2.write(v)
        f2.write('\n')
for tt in test_l:
    with open(r"G:\second_datasets\jpg\JM_test.txt",'a') as f3:
        f3.write(tt)
        f3.write('\n')

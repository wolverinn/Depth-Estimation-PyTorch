import os
import shutil
import pickle
import random

for folder in os.listdir("./basements"):
    for file in os.listdir("./basements/{}".format(folder)):
        if file[0] == "r":
            shutil.move("./basements/{}/{}".format(folder,file),"./nyuv2/train_rgb/{}".format(file))
        elif file[0] == "d":
            shutil.move("./basements/{}/{}".format(folder,file),"./nyuv2/train_depth/{}".format(file))
        else:
            continue

name_map = {}
for i in range(3):
    with open("INDEX{}.txt".format(str(i+1)),'r') as f:
        for line in f:
            if line[0] == "d":
                temp_d = line.strip('\n')
            elif line[0] == "r":
                if line.strip('\n') in os.listdir('./nyuv2/train_rgb'):
                    if temp_d in os.listdir('./nyuv2/train_depth'):
                        name_map[line.strip('\n')] = temp_d
            else:
                continue

fpkl = open("./nyuv2/index.pkl",'wb')
pickle.dump(name_map,fpkl)

train_dict = pickle.load(open("./nyuv2/index.pkl",'rb'))
test_dict = {}
train_r_list = list(train_dict.keys())
test_r_list = random.sample(train_r_list,600)
for r_file in test_r_list:
    shutil.move("./nyuv2/train_rgb/{}".format(r_file),"./nyuv2/test_rgb/{}".format(r_file))
    d_file = train_dict[r_file]
    shutil.copy("./nyuv2/train_depth/{}".format(d_file),"./nyuv2/test_depth/{}".format(d_file))
    test_dict[r_file] = d_file
    train_dict.pop(r_file)

test_pkl = open("./nyuv2/index2.pkl",'wb')
train_pkl = open("./nyuv2/index1.pkl",'wb')
pickle.dump(train_dict,train_pkl)
pickle.dump(test_dict,test_pkl)
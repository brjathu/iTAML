import os
import random
import numpy as np

all_paths = os.listdir("../../Datasets/MS1M/imgs/")
# print(all_paths)

folders = 0

train_imgs_all = []
val_imgs_all = []

for p in all_paths:
#     path_p = "../../Datasets/MS1M/10K/"+ p
    path_p = "../../Datasets/MS1M/imgs/"+ p
    imgs_path = os.listdir(path_p)
    if len(imgs_path)>45:
        folders += 1
        train_imgs = imgs_path[:30]
        val_imgs = imgs_path[30:45]
        for i in train_imgs:
            full_path = p + "/" + i
            train_imgs_all.append([full_path, int(p)])

        for i in val_imgs:
            full_path = p + "/" + i
            val_imgs_all.append([full_path, int(p)])
#     else:
#         r = len(imgs_path)//3
#         train_imgs = imgs_path[:r]
#         val_imgs = imgs_path[r:]
#         for i in train_imgs:
#             full_path = p + "/" + i
#             train_imgs_all.append([full_path, int(p)])

#         for i in val_imgs:
#             full_path = p + "/" + i
#             val_imgs_all.append([full_path, int(p)])
        


    
    if(folders>=10000):
        break

print(folders)

train_imgs_all = np.array(train_imgs_all)
val_imgs_all = np.array(val_imgs_all)


print(train_imgs_all.shape)
print(val_imgs_all.shape)

np.save("train_imgs.npy", train_imgs_all)
np.save("val_imgs.npy", val_imgs_all)



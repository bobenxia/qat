import os
from shutil import copy
from tqdm import tqdm

def mkdir(path):
    base_path='/data/xiazheng/tiny-imagenet-200/val/'
    path=os.path.join(base_path,path)
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path) 
 
        print (path+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print (path+' 目录已存在')
        return False


def rewrite_val(dir):
    with open(os.path.join(dir, 'val_annotations.txt'), 'r') as f:
        data = f.readlines()
    
    img_list = [] # 存放图片
    dir_list = [] # 图片对应标签的名称

    for i in range(len(data)):
        img_list.append((data[i].split('\t'))[0])
        dir_list.append((data[i].split('\t'))[1])

    return img_list, dir_list
 
img_name, file_name = rewrite_val('/data/xiazheng/tiny-imagenet-200/val')

print(len(img_name), len(file_name))
for i in range(len(file_name)):
	mkdir(file_name[i])

from_path = "/data/xiazheng/tiny-imagenet-200/val/images"
to_path = "/data/xiazheng/tiny-imagenet-200/val"
for i in tqdm(range(len(img_name))):
    copy(os.path.join(from_path,img_name[i]),os.path.join(to_path,file_name[i]))
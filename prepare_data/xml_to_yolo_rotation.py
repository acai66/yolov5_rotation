#!/usr/bin/env python
# coding: utf-8

# ## convert xml to [yolov5-rotation](https://github.com/acai66/yolov5_rotation) format. [旋转版yolov5](https://github.com/acai66/yolov5_rotation)标签格式转换

# In[1]:


import xml.etree.ElementTree as ET
from tqdm import tqdm # pip install tqdm
import os
import math


# In[2]:


workdir = './data_wire3/' # datasets root path. 数据集路径
images_dir = os.path.join(workdir, 'all_images') # images path. 图像路径
labels_dir = os.path.join(workdir, 'all_labels') # xml labels path. xml标签路径
# yolov5_all_images = os.path.join(workdir, 'yolov5_all_images') # all images for yolov5 rotation. 转换后旋转版yolov5可用的图像路径
yolov5_all_labels = os.path.join(workdir, 'yolov5_all_labels') # all labels for yolov5 rotation. 转换后旋转版yolov5可用的txt标签路径
for d in [yolov5_all_labels]:
    if not os.path.exists(d):
        os.mkdir(d)


# In[3]:


all_files = [i for i in os.listdir(labels_dir) if i[-4:] == '.xml']


# In[4]:


print('labels count: ', len(all_files))


# In[5]:


keep_class_names = [  ] # auto scan if blank. 如果留空，会自动扫描类别


# In[6]:


class_names = dict(zip(keep_class_names, range(len(keep_class_names))))


# ## convert. 转换

# In[7]:


auto_scan = False
class_index = 0
if len(class_names) == 0:
    auto_scan = True
    print('Auto scan classnames enabled.')

for file in tqdm(all_files):
    file_path = os.path.join(labels_dir, file)
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    img_size = root.find('size')
    width = float(img_size.find('width').text)
    height = float(img_size.find('height').text)
    
    objs = root.findall('object')
    
    with open(os.path.join(yolov5_all_labels, file[:-4] + '.txt'), 'w+') as f:
        for obj in objs:
            name = obj.find('name').text.strip()
            if name not in class_names.keys():
                if auto_scan:
                    class_names[name] = class_index
                    class_index = class_index + 1
                else:
                    continue
            rbb = obj.find('robndbox')
            if not rbb:
                print('no robndbox in %s' % (file_path))
            cx = float(rbb.find('cx').text)
            cy = float(rbb.find('cy').text)
            w = float(rbb.find('w').text)
            h = float(rbb.find('h').text)
            angle = float(rbb.find('angle').text)
            
            degree = round(angle / math.pi * 180)
            if h > w:     # swap w,h if w is not longside. 宽不是长边时，交换宽高
                w, h = h, w
                if degree < 90:
                    degree = degree + 90
                else:
                    degree = degree - 90
            cv_degree = 180 - degree     # opencv angle format. opencv格式角度
            if cv_degree == 180:
                cv_degree = 0
            assert cv_degree >= 0 and cv_degree < 180
            
            f.write('{} {} {} {} {} {}\n'.format(class_names[name], cx/width, cy/height, w/width, h/width, cv_degree))
    # break


# In[8]:


print('class_names: ', class_names)


# In[9]:


sorted_keys = [i[0] for i in sorted(class_names.items(), key = lambda kv:(kv[1], kv[0]))]


# In[10]:


print('names: [ "{}" ]'.format('", "'.join(sorted_keys)))


# ## split datasets for yolov5. 将转换格式后的数据集按yolov5方式组织

# In[11]:


import random
import shutil


# In[12]:


imgages = os.path.join(workdir, 'images')
labels = os.path.join(workdir, 'labels')
train_img = os.path.join(imgages, 'train')
val_img = os.path.join(imgages, 'val')
train_label = os.path.join(labels, 'train')
val_label = os.path.join(labels, 'val')
for d in [imgages, labels, train_img, val_img, train_label, val_label]:
    if not os.path.exists(d):
        os.mkdir(d)


# In[13]:


all_txts = [i for i in os.listdir(yolov5_all_labels) if i[-4:] == '.txt']


# In[14]:


random.shuffle(all_txts) # shuffle data. 随机打乱顺序


# ## split (train, test) rate. 随机划分数据集，比例

# In[15]:


train_factor = 0.8


# In[16]:


inds = int(train_factor * len(all_txts))


# In[17]:


print('all count: ', len(all_txts))
print('train count: ', inds)
print('test count: ', len(all_txts) - inds)


# In[18]:


train_txts = all_txts[:inds]
val_txts = all_txts[inds:]


# ## copy images and labels to train/val dirs, only run once. 只执行一次

# In[19]:


run = True


# In[20]:


if run:
    for txt in train_txts:
        shutil.copyfile(os.path.join(yolov5_all_labels, txt), os.path.join(train_label, txt))
        src = os.path.join(images_dir, txt[:-4] + '.jpg')
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(train_img, txt[:-4] + '.jpg'))
        else:
            shutil.copyfile(os.path.join(images_dir, txt[:-4] + '.png'), os.path.join(train_img, txt[:-4] + '.png'))


# In[21]:


if run:
    for txt in val_txts:
        shutil.copyfile(os.path.join(yolov5_all_labels, txt), os.path.join(val_label, txt))
        src = os.path.join(images_dir, txt[:-4] + '.jpg')
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(val_img, txt[:-4] + '.jpg'))
        else:
            shutil.copyfile(os.path.join(images_dir, txt[:-4] + '.png'), os.path.join(val_img, txt[:-4] + '.png'))


# In[ ]:





import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset,random_split
from torchvision import transforms as T
from PIL import Image
import torch
import matplotlib.pyplot as plt
torch.manual_seed(45)


class MyDataSet(Dataset):
    def __init__(self, image_dir, label_dir, col_name, transform = None):
        self.image_dir = image_dir
        self.transform = transform
        self.label_dir = label_dir
        self.col_name = col_name
        self.df = pd.read_excel(label_dir)
        self.image_ids = list(self.df['newname'])
        self.image_label = self.__getLabel__()


    def __len__(self):
        return len(self.image_ids)


    def __getLabel__(self):
        imageLabels = {}
        for index, row in self.df.iterrows():
            imageLabels[str(row['newname'])] = row[self.col_name]
        # print(imageLabels)
        return imageLabels


    def __getitem__(self,index):
        image_id = self.image_ids[index]
        img_path = os.path.join(self.image_dir, str(image_id))
        img = Image.open(img_path).convert('RGB')
        label = self.image_label[str(image_id)]
        if self.transform is not None:
            img = self.transform(img)
        return img, label, image_id


def dataSplit(dataset, train_size=0.8, test_size=0.2, valid_size=None):
    """划分数据集"""
    len_data = dataset.__len__()
    if valid_size == None:

        if train_size + test_size != 1:
            raise ValueError("train_size + test_size != 1")

        train_size = int(len_data * train_size)
        test_size = len_data - train_size
        return random_split(dataset, [train_size, test_size])
    else:
        if train_size + test_size + valid_size != 1:
            raise ValueError("train_size + test_size + valid_size != 1")
        train_size = int(len_data * train_size)
        test_size = int(len_data * test_size)
        valid_size = len_data - train_size - test_size
        return random_split(dataset, [train_size, test_size, valid_size])


if __name__=='__main__':
    img_dir = '../Data/sec_mix_4_classes4'
    label_dir = '../Data/所有图像对应表_有标签.xlsx'

    transform = T.Compose([
                # T.Resize([300,400]),
                T.CenterCrop([400,760]),
                # T.ToTensor(),
                # T.Normalize(mean=[0.485, 0.456, 0.406,],
                #             std=[0.229, 0.224, 0.225])
            ])

    dataset = MyDataSet(img_dir,label_dir,col_name='label',transform = transform)

    train_data, test_data = dataSplit(dataset)
    # test_data = dict(test_data)

    test_dic = {}
    for _,label,img  in test_data:
        test_dic[img] = label
    print(test_dic)
    df = pd.DataFrame.from_dict(test_dic,orient='index',columns=['label'])
    df = df.reset_index().rename(columns = {'index':'imgname'})
    df.to_excel('testset.xlsx')


    img, label,img_name = train_data[160]
    print(type(img),label,img_name)
    print(img)
    # img = img.numpy().transpose((1, 2, 0))
    plt.figure()
    plt.imshow(img)
    plt.show()



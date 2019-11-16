from torch.utils.data import Dataset
import os
import cv2
import numpy as np


class MyDataset(Dataset):
    def __init__(self, train_path, transform=None):
        self.images = os.listdir(train_path + '/last')
        self.labels = os.listdir(train_path + '/last_msk')
        assert len(self.images) == len(self.labels), 'Number does not match'
        self.transform = transform
        self.images_and_labels = []    # 存储图像和标签路径
        for i in range(len(self.images)):
            self.images_and_labels.append((train_path + '/last/' + self.images[i], train_path + '/last_msk/' + self.labels[i]))

    def __getitem__(self, item):
        img_path, lab_path = self.images_and_labels[item]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        lab = cv2.imread(lab_path, 0)
        lab = cv2.resize(lab, (224, 224))
        lab = lab / 255    # 转换成0和1
        lab = lab.astype('uint8')    # 不为1的全置为0
        lab = np.eye(2)[lab]    # one-hot编码
        lab = np.array(list(map(lambda x: abs(x-1), lab))).astype('float32')   # 将所有0变为1（1对应255， 白色背景），所有1变为0（黑色，目标）
        lab = lab.transpose(2, 0, 1)  # [224, 224, 2] => [2, 224, 224]
        if self.transform is not None:
            img = self.transform(img)
        return img, lab

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    img = cv2.imread('data/train/last_msk/150.jpg', 0)
    img = cv2.resize(img, (16, 16))
    img2 = img/255
    img3 = img2.astype('uint8')
    hot1 = np.eye(2)[img3]
    hot2 = np.array(list(map(lambda x: abs(x-1), hot1)))
    print(hot2.shape)
    print(hot2.transpose(2, 0, 1))

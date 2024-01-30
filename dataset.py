import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Resize


class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super(MyDataset, self).__init__()
        self.data_list = []
        with open(data_path, encoding='utf-8') as f:
            for line in f.readlines():
                image_path = line.replace('\n', '')
                label = 0
                self.data_list.append([image_path, label])
        self.transform = transform
        self.fore_num_sum = 0
        self.back_num_sum = 0

    def __getitem__(self, index):
        image_path, label = self.data_list[index]
        # print('------------------')
        # print('1. image path:' image_path)
        image = cv2.imread(image_path)
        # self.show_image('img', image)
        if self.transform is not None:
            # print(image.shape)
            image = self.transform(image)
        # self.show_image('img', image)

        image = image.astype('float32')
        image = torch.tensor(np.array(image))
        # print('4.image shape after change to tensor:', image.shape)
        image = image.unsqueeze(0)  # CHW
        C, H, W = image.shape
        image = image.expand(3, H, W)
        # print('5. image shape after add channel in axis 0:', image.shape)
        fore_num = float((image == 0).sum())
        back_num = float((image == 255).sum())
        self.fore_num_sum = self.fore_num_sum + fore_num
        self.back_num_sum = self.back_num_sum + back_num

        # print(f'- fore_num = {fore_num}, back_num = {back_num}')
        # print(f'- fore_num = {self.fore_num_sum}, back_num = {self.back_num_sum}, fore_ratio = {self.fore_num_sum/self.back_num_sum}')
        # print('------------------')
        label = int(label)
        return image, label

    def __len__(self):
        return len(self.data_list)

    def show_image(self, name, img):
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()


class Compose():
    def __init__(self):
        pass

    def __call__(self, image):
        # print('2. enter image transform')
        imgGray = self.convert_gray_scale(image)
        # print(imgGray.shape)
        imgBW = self.convert_binary(imgGray)
        imgBW = 255 - imgBW
        transformed_img = cv2.resize(imgBW, (640, 640))
        # resize_operator = Resize(size=(640, 640))
        # transformed_img = resize_operator(imgBW)
        imgBW1 = self.convert_binary(transformed_img)
        imgBW1 = 255 - imgBW1
        # print('3. image path after transform:', imgBW1.shape)

        return imgBW1

    def convert_gray_scale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def convert_binary(self, imgGray):
        ret, imgBW = cv2.threshold(imgGray, 230, 255, cv2.THRESH_BINARY_INV)
        return imgBW


def get_dataset(mode='train'):
    assert mode in ['train', 'val']
    transforms = Compose()
    if mode == 'train':
        dataset = MyDataset('train_data.txt', transforms)
    else:
        dataset = MyDataset('train_data.txt', transforms)

    return dataset


def get_dataloader(dataset, mode='train', batch_size=16):
    assert mode in ['train', 'val']
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=20,
                            # shuffle =(mode=='train')
                            shuffle=True)

    return dataloader


if __name__ == '__main__':
    train_dataset = get_dataset(mode='train')
    val_dataset = get_dataset(mode='val')
    dataloader = get_dataloader(train_dataset, batch_size=2)

    # print(len(train_dataset))
    # print(len(val_dataset))

    image, label = train_dataset[0]
    # print(image)
    image, label = val_dataset[0]
    # print(image)

    count = 1
    for imgs, labels in dataloader:
        print(imgs.shape)
        # print(labels.shape)
        # break
        # print(count
        # count = count + 1

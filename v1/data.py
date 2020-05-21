import os
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

class SalObjDataset(data.Dataset):
    def __init__(self, image_path, gt_path, trainsize):
        self.images = [image_path + f for f in os.listdir(image_path) if f.endswith('.jpg')]
        self.gts = [gt_path + f for f in os.listdir(gt_path) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image, gt = self.cv_random_flip(image, gt)
        image, gt = self.cv_random_rotate(image, gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        return image, gt


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def gray_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def cv_random_flip(self, img, gt):
        if np.random.randint(2)==0:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
        return img, gt

    def cv_random_rotate(self, img, gt):
        rotate_degree = np.random.random() * 2 * 10 - 10
        img = img.rotate(rotate_degree, Image.BILINEAR)
        gt = gt.rotate(rotate_degree, Image.NEAREST)
        return img, gt

    def __len__(self):
        return self.size


def get_loader(image_path, gt_path, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):

    dataset = SalObjDataset(image_path, gt_path, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_path, testsize):
        self.images = [image_path + f for f in os.listdir(image_path) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        img_size = (image.size[1], image.size[0])
        image = self.img_transform(image).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, img_size, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')



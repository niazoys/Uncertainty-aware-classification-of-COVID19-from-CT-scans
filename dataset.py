
import os
import sys
import h5py
import imageio
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

class OCTDataset(Dataset):
    
    def __init__(self,path:str,shape=None,training=True):
        self.path=path
        self.training=training
        self.shape=shape

        self.trans_train = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Resize(size=self.shape),
                                  transforms.ColorJitter(brightness=0.25,contrast=0.25),
                                  transforms.RandomResizedCrop(size=self.shape),
                                  transforms.RandomHorizontalFlip(), 
                                  transforms.RandomVerticalFlip(),
                                  transforms.RandomRotation(20), 
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5],std=[0.5])])

        self.trans_val_test = transforms.Compose([  transforms.ToPILImage(),
                                                    transforms.Resize(size=self.shape),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.5],std=[0.5])])     

    def __len__(self):
        self.files = os.listdir(self.path)    
        return len(self.files)

    def __getitem__(self, idx:int):
        img =np.array(imageio.imread( os.path.join(self.path,self.files[idx])))
        
        if self.training:
            img=self.trans_train(img)
        else:
            img=self.trans_val_test(img)
        
        # repeat same thing in all three channels
        img=img.squeeze(0)
        img=img.repeat(3, 1, 1)

        if self.files[idx][:2]=="CN":
            label=0
        elif self.files[idx][:2]=="DM":
            label=1
        elif self.files[idx][:2]=="DR":
            label=2
        elif self.files[idx][:2]=="NO":
            label=3
        else:
            sys.exit("Improper naming of input file. Class couldn't be resolved.")
        
        return img,label

class COVIDDataset(torchvision.datasets.ImageFolder):
    def __init__(self,root,shape,train=True,transform=None):
        super(COVIDDataset, self).__init__(root, transform)

        if train:
            self.transform = transforms.Compose([
                                                transforms.Resize(size=(shape,shape)),
                                                transforms.ColorJitter(brightness=0.25,contrast=0.25),
                                                transforms.RandomResizedCrop(size=(shape,shape)),
                                                transforms.RandomHorizontalFlip(), 
                                                transforms.RandomVerticalFlip(),
                                                transforms.RandomRotation(20), 
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.4914],std=[0.2023])])
        else:
            self.transform = transforms.Compose([
                                                transforms.Resize(size=(shape,shape)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.4914],std=[0.2023])])    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.transform(self.loader(path))
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

class H5Dataset(Dataset):
    def __init__(self, path,shape,train=True):
        self.shape=(shape,shape)
        self.file_path = path
        self.train = train
        self.data = np.array(h5py.File(self.file_path, 'r')["ct_slices"]).astype(np.uint8)
        self.label = np.array(h5py.File(self.file_path, 'r')["slice_class"]).astype(np.int64)
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file['slice_class'])

        self.trans_train = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Resize(size=self.shape),
                                  transforms.ColorJitter(brightness=0.25,contrast=0.25),
                                  transforms.RandomResizedCrop(size=self.shape),
                                  transforms.RandomHorizontalFlip(), 
                                  transforms.RandomVerticalFlip(),
                                  transforms.RandomRotation(20), 
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5],std=[0.5])])

        self.trans_val_test = transforms.Compose([  transforms.ToPILImage(),
                                                    transforms.Resize(size=self.shape),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.5],std=[0.5])])     

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        
        if self.train:
            img=self.data[index]
            img=self.trans_train(img)
        else:
            img=self.trans_val_test(self.data[index])
        
        # repeat same thing in all three channels
        img=img.squeeze(0)
        img=img.repeat(3, 1, 1)

        return  img,self.label[index]


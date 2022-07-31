
import os
import imageio
import numpy as np
import torchvision.transforms as transforms
import torchvision
from torch.utils.data.dataset import Dataset
import sys
import h5py

class oct_dataset(Dataset):
    
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
        # img=(img/255).astype(np.uint8)
        if self.training:
            img=self.trans_train(img)
        else:
            img=self.trans_val_test(img)

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



class H5Dataset(Dataset):
    def __init__(self, path,shape,train=True):
        self.shape=(shape,shape)
        self.file_path = path
        self.train = train
        self.data = np.array(h5py.File(self.file_path, 'r')["ct_slices"]).astype(np.uint8)
        self.label = np.array(h5py.File(self.file_path, 'r')["slice_class"]).astype(np.int)
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

        return  img,self.label[index]

   
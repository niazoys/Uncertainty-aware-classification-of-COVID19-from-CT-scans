import os
import torch
import utils
import logging
import argparse
import numpy as np
from tqdm import tqdm
from torch import optim
import torchvision
from model import vgg,vgg_adf
import distutils.dir_util
import torch.nn as nn
import dataset
from torch.utils.data import DataLoader
from adf_blocks import SoftmaxHeteroscedasticLoss

def train(net,output,device,epochs,criterion,train_dataset,val_dataset,batch,lr):
 
    n_train, n_val = len(train_dataset), len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True,num_workers=0)
  
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True,num_workers=0)
   
    optimizer = optim.Adam(net.parameters(), lr=1e-5*lr)
    
    logging.info("Training Set lenghth: "+str(n_train)+" Validation Set length: "+str(n_val))
    logging.info("Learning Rate: "+str(1e-5*lr))
    logging.info("Batch Size: "+str(batch))
    logging.info("Epochs: "+str(epochs))
    n_batch_train=n_train/batch
    n_batch_val=n_val/batch
    train_acc, val_acc,train_loss,val_loss= [],[],[], []
    best_accuracy = -1

    for epoch in range(epochs):
        logging.info(f'epoch {epoch + 1}/{epochs}')
        net.train()
        epoch_loss = 0
        correct_train = 0
        
        with tqdm(total=n_train, desc=f'epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs, label = batch[0], batch[1]
                imgs = (imgs.to(device=device))
                label = (label.to(device=device))
                preds = net(imgs) 
                loss = criterion(preds, label)
                if loss is not None:
                    epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})             
                _,preds = torch.max(preds[0], dim=1)
                correct_train+=torch.sum(preds==label.squeeze()).item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(imgs.shape[0])
                
                
        #Training matrices
        train_accuracy = 100 * (correct_train / n_train)
        train_acc.append(train_accuracy)
        train_loss.append(epoch_loss/n_batch_train)
        logging.info(f'Training Accuracy: {train_accuracy} Avg training batch loss: {epoch_loss/n_batch_train}')

        #Validation matrices
        correct_val,validation_loss,y_true,y_pred= utils.eval_net(net,val_loader,criterion,device)
        val_accuracy = 100*correct_val/n_val
        val_acc.append(val_accuracy)
        val_loss.append(validation_loss/n_batch_val)
        logging.info(f'Validation Accuracy: {val_accuracy}  Avg validation batch loss: {validation_loss/n_batch_val}')

        if epoch>0:
            if val_acc[-1]>best_accuracy :
                os.remove(output + f'vgg11.pth')
                torch.save(net.state_dict(), output + f'vgg11.pth')
                best_accuracy = val_acc[-1]
                logging.info(f'checkpoint {epoch} saved !')
        else:
            torch.save(net.state_dict(), output + f'vgg11.pth')
    
        #Save the model and training,validation Accuracy,dice score
        np.save(output+'traning_acc.npy',train_acc)
        np.save(output+'val_acc.npy',val_acc)
        np.save(output+'training_loss.npy',train_loss)
        np.save(output+'val_loss.npy',val_loss)
    return net

def get_args():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-o', '--output', type=str, default='results/vgg11_tf_uq/', dest='output')
    
    parser.add_argument('-e', '--epochs', type=int, default=30, dest='epochs')
    
    parser.add_argument('-b', '--batch', type=int, default=10, dest='batch')
    
    parser.add_argument('-l', '--learning', type=int, default=100, dest='lr')
    
    parser.add_argument('-is', '--in_shape', type=int, default=128, dest='in_shape')
    
    # parser.add_argument('-f', '--load', type=str, default='results/vgg19_bs/epoch_vgg19.pth', dest='load')    
   
    parser.add_argument('-f', '--load', type=str, default=None, dest='load')

    parser.add_argument('-ad', '--adf', type=str, default=True, dest='adf')
   
    return parser.parse_args()

if __name__ == '__main__':
    
    # Setup the logging level
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # parse arguments
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info('Using device'+str(device))

    if args.adf:    
        # Define mini variance and noise variance and other parameters 
        min_variance,noise_variance= 1e-9,1e-3
        dropout_prob=0.2
        # Get the network and initialize the weights
        net=vgg_adf(variant='vgg11',num_classes=2,dropout_prob=dropout_prob,min_variance=min_variance,noise_variance=noise_variance)
        net=utils.init_params(net)
        criterion = SoftmaxHeteroscedasticLoss(min_variance=min_variance)
    else:
        net=vgg(variant='vgg11',num_classes=2)
        criterion= nn.CrossEntropyLoss()
    
    # make output directory if not exist
    if not os.path.isdir(args.output):
        distutils.dir_util.mkpath(args.output)
    
    # Load previously saved weight
    if args.load != None:        
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')
    
    # Transfer the model to cuda 
    net.to(device=device)
    
    #Get the dataset (Pulmonary Dataset for model pre training)
    train_dataset_pul=dataset.H5Dataset(path='pulmonary_data/train_data.h5',train=True,shape=args.in_shape)
    val_dataset_pul=dataset.H5Dataset(path='pulmonary_data/val_data.h5',train=False,shape=args.in_shape)

    #Get the COVID dataset
    train_dataset_covid=dataset.COVIDDataset(root='covid_data/train',shape=args.in_shape,train=True)
    val_dataset_covid=dataset.COVIDDataset(root='covid_data/val',shape=args.in_shape,train=False)
    
    # # pretrain on pulmonary dataset        
    net = train(net = net,output = args.output,device = device,criterion=criterion,epochs = args.epochs,train_dataset=train_dataset_pul,val_dataset=val_dataset_pul,batch = args.batch,lr = args.lr)
    
    # train on covid dataset
    _   = train(net = net,output = args.output,device = device,criterion=criterion,epochs = args.epochs,train_dataset=train_dataset_covid,val_dataset=val_dataset_covid,batch=args.batch,lr=args.lr)
    
import os
import torch
import utils
import logging
import argparse
import adf_blocks
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from model import vgg,vgg_adf
import distutils.dir_util
import torch.nn.functional as F
from dataset import COVIDDataset


def predict(net,output,device,batch_size,use_adf,use_mcdo):
 
    test_dataset = COVIDDataset("test/",training=False,shape=(128,128))
 
    test_loader = torch.utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=3)

    y_true,y_pred,correct=[],[],0
    test_loss,brier_score,nll=0,0,0
    output_variance=None
    
    net.eval()
    with torch.no_grad():
        for batch in enumerate(tqdm(test_loader)):
            imgs, label = batch[1][0], batch[1][1]
            imgs = (imgs.to(device=device))
            label = (label.to(device=device))
            mean, data_var, model_var = compute_preds(net, imgs, use_adf, use_mcdo)
            
            if data_var is not None and model_var is not None:
                outputs_variance = data_var + model_var
            elif data_var is not None:
                outputs_variance = data_var
            elif model_var is not None:
                outputs_variance = model_var + args.tau
            
            one_hot_label=F.one_hot(label)

            if outputs_variance is not None:
                batch_nll = -utils.compute_log_likelihood(mean, one_hot_label, outputs_variance)
                # Sum along batch dimension
                neg_log_likelihood += torch.sum(batch_nll, 0).cpu().numpy().item()

            preds = net(imgs) 
            preds = F.softmax(preds,dim=1)
            _,preds = torch.max(preds, dim=1)
            correct+=correct.sum(preds==label).item()
            for i in range(len(preds)):
                y_true.append(label.cpu().numpy()[i])
                y_pred.append(preds.cpu().numpy()[i])
        
    accuracy = 100*correct/len(test_dataset)
    logging.info(f'Test Accuracy: {accuracy}')
    prec,recall=utils.stats(torch.tensor(y_pred),torch.tensor(y_true),n_classes=4,output_path=output)
    np.save(output+'y_pred.npy',y_pred)
    np.save(output+'y_true.npy',y_true)
    #Save the stats
    file = open(output+'stat.txt','w')
    file.write('Precision = %f\n'%(prec))
    file.write('Recall = %f\n'%(recall))
    file.close()

def compute_preds(net, inputs, min_variance=1e-4, use_adf=False, use_mcdo=False):
    
    model_variance = None
    data_variance = None
    
    def keep_variance(x, min_variance):
        return x + min_variance

    var_fun = lambda x: keep_variance(x, min_variance=min_variance)
    softmax = nn.Softmax(dim=1)
    adf_softmax = adf_blocks.Softmax(dim=1, keep_variance_fn=var_fun)
    
    net.eval()

    if use_mcdo:
        # Unfreeze the dropout layers
        net = utils.freeze_unfreeze_dropout(net, True)
        #  make prediction n times 
        outputs = [net(inputs) for i in range(args.num_samples)]
        
        if use_adf:
            outputs = [adf_softmax(*outs) for outs in outputs]
            outputs_mean = [mean for (mean, var) in outputs]
            data_variance = [var for (mean, var) in outputs]
            data_variance = torch.mean(torch.stack(data_variance), dim=0)
        else:
            outputs_mean = [softmax(outs) for outs in outputs]
        
        outputs_mean = torch.stack(outputs_mean)
        # Compute the prediction variance (varinace of n number of prediction trail mean) 
        model_variance = torch.var(outputs_mean, dim=0)
        # Compute MCDO prediction (Average of n number of prediction trail mean)
        outputs_mean = torch.mean(outputs_mean, dim=0)
    else:
        outputs = net(inputs)
       
        if use_adf:
            outputs_mean, data_variance = adf_softmax(*outputs)
        else:
            outputs_mean = outputs
    
    #refreeze the dropout layers
    net = utils.freeze_unfreeze_dropout(net, False)
    
    return outputs_mean, data_variance, model_variance

def get_args():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-b', '--batch', type=int, default=25, dest='batch')
    
    parser.add_argument('-n', '--network', type=str, default='resnet34', dest='net')
    
    parser.add_argument('-f', '--load', type=str, default='results/vgg19_bs/', dest='load')  
    
    parser.add_argument('-ad', '--adf', type=str, default=True, dest='adf')
    
    parser.add_argument('-m', '--mcdo', type=str, default=True, dest='mcdo')
        
    return parser.parse_args()

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info('Using device'+str(device))
   
    net=vgg(n_classes=4,pretrained=False)
   

    #laod weights
    if args.load != None:        
        net.load_state_dict(torch.load(args.load+'epoch_vgg19.pth', map_location=device))
        logging.info(f'Model loaded from {args.load}')
    
    # create directory if not already exists
    output =args.load
   
    if not os.path.exists(output):
        distutils.dir_util.mkpath(output)

    net.to(device=device)
            
    predict(net = net,output = output,device = device,batch = args.batch)
        

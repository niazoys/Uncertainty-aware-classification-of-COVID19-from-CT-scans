
import torch
import adf_blocks
import numpy as np
import seaborn as sn
from tqdm import tqdm
from numbers import Number
import torch.nn.init as init
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchmetrics.functional import recall,precision,confusion_matrix


def stats(preds,gt,n_classes,output_path):
    ''' This method calculates the stats'''
    rec=recall(preds, gt, average='macro', num_classes=n_classes,mdmc_average='global')
    prec = precision(preds, gt, average='macro', num_classes=n_classes,mdmc_average='global')
    cm=confusion_matrix(preds, gt,num_classes=n_classes)
    plt.figure(figsize = (12,7))
    sn.heatmap(cm, annot=True)
    plt.savefig(output_path + 'output.png')   
    return prec,rec

def eval_net(net,loader,criterion,device):
    '''Forward passs with no grad'''
    
    net.eval()
    total = 0
    n_sample= 0
    loss_=0
    y_true,y_pred=[],[]
    for batch in enumerate(tqdm(loader)):
        imgs, label = batch[1][0], batch[1][1]
        imgs = (imgs.to(device=device))
        label = (label.to(device=device))
        with torch.no_grad():
            preds = net(imgs) 
        loss = criterion(preds, label)
        loss_ += loss.item()  
      
        if isinstance(preds,tuple):
            _,preds = torch.max(preds[0], dim=1)
        else:
            _,preds = torch.max(preds, dim=1)

        total+=torch.sum(preds==label.squeeze()).item()
        n_sample+=len(label)
        for i in range(len(preds)):
            y_true.append(label.cpu().numpy()[i])
            y_pred.append(preds.cpu().numpy()[i])

    return total,loss_,y_true,y_pred

def display_images(imageSet,mutiple=True):
    '''This method takes a List of image and shows them in a thumbnil fashion'''
    import matplotlib.pyplot ; 

    if mutiple:
        # Define the grid size of the viewing
        height=np.ceil(np.sqrt(imageSet.shape[2])).astype(int)
        width=np.ceil((imageSet.shape[2])/height +  1)

        #Define the size each individule figure in the grid
        fig=matplotlib.pyplot.figure(figsize=(15,15))

        #Go through all the files and put them in the figure
        for idx in range(imageSet.shape[0]):
            fig.add_subplot(height,width,idx+1)
            matplotlib.pyplot.axis('off')
            matplotlib.pyplot.imshow(imageSet[idx,:,:] , cmap='gray')
        matplotlib.pyplot.show()
    else:
        matplotlib.pyplot.axis('off')
        matplotlib.pyplot.imshow(imageSet, cmap='gray')
        matplotlib.pyplot.show()

def init_params(net):
    ''' Init layer parameters '''
    for m in net.modules():
        if isinstance(m, adf_blocks.Conv2d):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, adf_blocks.BatchNorm2d):
            init.normal_(m.weight)
            init.constant_(m.bias, 0)
        elif isinstance(m, adf_blocks.Linear):
            init.kaiming_normal_(m.weight)
            init.constant_(m.bias, 0)
    return net

def brierscore(input, target_one_hot): 
    
    pt = F.softmax(input, dim = 1)
    squared_diff = torch.sum((target_one_hot.squeeze() - pt) ** 2, axis = 1)
    sum_squared_diff = torch.sum(squared_diff)
    loss = sum_squared_diff / (float(2*pt.shape[0])) #*2 for boundaries between [0,1]

    return loss

def keep_variance(x, min_variance):
        return x + min_variance

def normcdf(value, mu=0.0, stddev=1.0):
    sinv = (1.0 / stddev) if isinstance(stddev, Number) else stddev.reciprocal()
    return 0.5 * (1.0 + torch.erf((value - mu) * sinv / np.sqrt(2.0)))

def _normal_log_pdf(value, mu, stddev):
    var = (stddev ** 2)
    log_scale = np.log(stddev) if isinstance(stddev, Number) else torch.log(stddev)
    return -((value - mu) ** 2) / (2.0*var) - log_scale - np.log(np.sqrt(2.0*np.pi))

def normpdf(value, mu=0.0, stddev=1.0):
    return torch.exp(_normal_log_pdf(value, mu, stddev))

def freeze_unfreeze_dropout(net, training=True):
    """Set Dropout mode to train or eval."""
    for m in net.modules():
        if m.__class__.__name__.startswith('Dropout'):
            if training==True:
                m.train()
            else:
                m.eval()
    return net 

def resize2D_as(inputs, output_as, mode="bilinear"):
    size_targets = [output_as.size(2), output_as.size(3)]    
    size_inputs = [inputs.size(2), inputs.size(3)]

    if all([size_inputs == size_targets]):
        return inputs  # nothing to do
    elif any([size_targets < size_inputs]):
        resized = F.adaptive_avg_pool2d(inputs, size_targets)  # downscaling
    else:
        resized = F.upsample(inputs, size=size_targets, mode=mode)  # upsampling

    # correct scaling
    return resized

def compute_log_likelihood(y_pred, y_true, sigma):
    dist = torch.distributions.normal.Normal(loc=y_pred, scale=sigma)
    log_likelihood = dist.log_prob(y_true)
    log_likelihood = torch.mean(log_likelihood, dim=1)
    return log_likelihood
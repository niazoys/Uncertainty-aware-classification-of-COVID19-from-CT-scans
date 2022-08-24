import os
import torch
import utils
import logging
import argparse
import adf_blocks
import model_adf,model_dropout,model_classic
from tqdm import tqdm
import distutils.dir_util
from dataset import COVIDDataset


def predict(net,device,batch_size,use_adf,use_mcdo):
 
    test_dataset = COVIDDataset("covid_data/test/",train=False,shape=128)
 
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()

    y_true,y_pred,correct=[],[],0
    test_loss,brier_score,nll=0,0,0
    outputs_variance=None
    total,total_model_variance=0,0
    net.eval()

    with torch.no_grad():
        for _,batch in enumerate(tqdm(test_loader)):
            imgs, label = batch[0], batch[1]
            imgs = (imgs.to(device=device))
            label = (label.to(device=device))
            means, data_var, model_var = compute_preds(net, imgs, use_adf, use_mcdo)
            
            if data_var is not None and model_var is not None:
                outputs_variance = data_var + model_var

            elif data_var is not None:
                outputs_variance = data_var
            elif model_var is not None:
                outputs_variance = model_var + 1e-5



            one_hot_label=torch.nn.functional.one_hot(label,num_classes=2)

            # Compute NLL
            if outputs_variance is not None:
                # batch_nll = -utils.compute_log_likelihood(means, one_hot_label, outputs_variance)
                # Sum along batch dimension
                # nll += torch.sum(batch_nll, 0).cpu().numpy().item()
                nll=0

            # Compute Brier score
            batch_brier_score=utils.brierscore(means,one_hot_label)
            # Sum along batch dimension
            brier_score += torch.sum(batch_brier_score, 0).cpu().numpy().item()
            
            # Test loss for evaluation
            loss = criterion(means, label)
            test_loss += loss.item()

            # Compute predictions and numer of correct predictions
            _, predicted = means.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            if use_mcdo:
                #total prediction varince (model variance)
                total_model_variance+=model_var[:,predicted.item()]

            # Keep the results for further use
            y_pred.append(predicted)
            y_true.append(label)
    
    # Compute stat for whole test set
    accuracy = 100.*correct/total
    brier_score = brier_score/total
    nll = nll/total
    
    return accuracy, brier_score, nll, total_model_variance/len(test_loader)

def compute_preds(net, inputs,  use_adf, use_mcdo , min_variance=1e-4):
    
    model_variance = None
    data_variance = None
    
    def keep_variance(x, min_variance):
        return x + min_variance

    var_fun = lambda x: keep_variance(x, min_variance=min_variance)
    softmax = torch.nn.Softmax(dim=1)
    adf_softmax = adf_blocks.Softmax(dim=1, keep_variance_fn=var_fun)

    if use_mcdo:
        # Unfreeze the dropout layers
        net = utils.freeze_unfreeze_dropout(net, True)
        #  make prediction n times 
        outputs = [net(inputs) for i in range(args.num_samples)]
        
        if use_adf:
            outputs = [adf_softmax(*outs) for outs in outputs]
            outputs_mean = [means for (means, _) in outputs]
            data_variance = [var for (_ , var) in outputs]
            data_variance = torch.mean(torch.stack(data_variance), dim=0) 
        else:
            outputs_mean = [softmax(outs) for outs in outputs]
        
        outputs_mean = torch.stack(outputs_mean)
        # Compute the prediction variance (varinace of n number of prediction trail means) 
        model_variance = torch.var(outputs_mean, dim=0)
        # Compute MCDO prediction (Average of n number of prediction trail means)
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
    
    parser.add_argument('-b', '--batch', type=int, default=1, dest='batch')
    
    parser.add_argument('-f', '--load', type=str, default='results/resnet18_adf_True_mcdo_True/', dest='load')  
    
    parser.add_argument('-ad', '--adf', type=str, default=True, dest='adf')
    
    parser.add_argument('-m', '--mcdo', type=str, default=False, dest='mcdo')

    parser.add_argument('-ns', '--num_samples', type=int, default=4, dest='num_samples')

    parser.add_argument('-mn', '--model_name', type=str, default='resnet18', dest='model_name')

    parser.add_argument('-dp', '--dropout', type=float, default=0.1, dest='dropout_prob')
        
    return parser.parse_args()

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info('Using device'+str(device))
   
    if args.adf:    
        # Define mini variance and noise variance and other parameters 
        min_variance,noise_variance= 1e-4,1e-4
        # Get the network and initialize the weights
        net_builder = getattr(model_adf, args.model_name)
        net = net_builder(num_classes=2,dropout_prob=args.dropout_prob,min_variance=min_variance,noise_variance=noise_variance)
        criterion = adf_blocks.SoftmaxHeteroscedasticLoss(min_variance=min_variance,num_class=2)
    elif args.mcdo:
        net_builder = getattr(model_dropout, args.model_name)
        net=net_builder(num_classes=4,dropout_prob=args.dropout_prob)
        criterion= torch.nn.CrossEntropyLoss()
    else:
        net_builder = getattr(model_classic, args.model_name)
        net=net_builder(num_classes=4,dropout_prob=args.dropout_prob)
        criterion= torch.nn.CrossEntropyLoss()
   
    #laod weights
    if args.load != None:        
        net.load_state_dict(torch.load(args.load+args.model_name+'.pth', map_location=device))
        logging.info(f'Model loaded from {args.load}')
    
    # create directory if not already exists
    output =args.load
   
    if not os.path.exists(output):
        distutils.dir_util.mkpath(output)

    net.to(device=device)
            
    accuracy, brier_score, nll,total_prediction_variance=predict(net = net,device = device,batch_size = args.batch,use_adf=args.adf,use_mcdo=args.mcdo)


    logging.info(f'Test Accuracy: {accuracy}')
    logging.info(f'Test Brier Score: {brier_score}')
    logging.info(f'Negativge Log Loss: {nll}')    
    logging.info(f'Total Prediction variance for the test set: {total_prediction_variance}')
  
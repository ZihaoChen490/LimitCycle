from Data_Utils import *
from Plot_Utils import *
from Math_Utils import *
import numpy as np
import math, torch,argparse
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.ion()
from VRNN_model import VRNN
import dgm
"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""





class vrnn:
    def __init__(self,Dataset,args):
        self.Dataset=Dataset
        self.steps=args.steps
        self.gap=args.gap
        self.lr=args.learning_rate
        self.weight_decay=args.weight_decay
        self.epoch=args.epoch
        self.InputSet,self.TargetSet=Id2Id(self.Dataset.train_dl,args.gap)
        self.NewIndex, self.TargetSet = np.loadtxt('NewIndex.txt'),np.loadtxt('TargetSet.txt')
        if self.NewIndex is None:
            self.NewIndex,self.TargetSet=ReIndex(Dataset.train_dl[:len(Dataset.train_dl)//10],args.gap,40)
    def train_vrnn(self,model,args):
        train_dl=self.Dataset.train_dl
        gap=self.gap
        steps=self.steps
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)
        regularization_loss,t = 0,len(train_dl)//steps
        loss0=0
        for epoch in range(self.epoch):
            for step in range(steps-1385):
                data = torch.from_numpy(np.vstack(
                    [self.TargetSet[self.NewIndex[step * t:(step + 1) * t].astype(np.int), :] for i in range(40)]).astype(np.float32))
                print('datashape is', data.shape)
                data = data.to(device)
                loss_res = []
                kld_loss_res = []
                nll_loss_res = []
                print('.........................................')
                print('Now we strat to train epoch:',epoch,'step is',step)
                model.zero_grad()
                inputs=data[step*t:(step+1)*t]
                print('inputs is',inputs.shape)
                kld_loss, nll_loss = model(inputs)
                loss = kld_loss + nll_loss
                # grad norm clipping, only in pytorch version >= 1.10
                loss.backward(retain_graph=True)
                loss_res.append(float(loss.detach().numpy()))
                kld_loss_res.append(float(loss.detach().numpy()))
                nll_loss_res.append(float(loss.detach().numpy()))
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                print('Train Epoch: {} \t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(epoch,sum(kld_loss_res),  sum(nll_loss_res)))
                print('Now the sum_loss=',sum(loss_res))
            torch.save(model, args.save_path)
            #torch.save(model.state_dict(), save_path)
            #model.eval()
            print('Model has been saved')
            if  abs(loss0-sum(loss_res))<0.0001:
                sample = model.sample(torch.tensor(28, device=device))
                plt.imshow(sample.to(torch.device('cpu')).numpy())
                plt.pause(1e-6)
                break
            else:
                loss0=sum(loss_res)

    def test(self,epoch):
        """uses test data to evaluate
        likelihood of the model"""
        mean_kld_loss, mean_nll_loss = 0, 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.Dataset.test_dl):
                data = data.to(device)
                data = data.squeeze().transpose(0, 1)
                data = (data - data.min()) / (data.max() - data.min())
                kld_loss, nll_loss, _, _ = model(data)
                mean_kld_loss += kld_loss.item()
                mean_nll_loss += nll_loss.item()

        mean_kld_loss /= len(self.dataset.test_dl)
        mean_nll_loss /= len(self.dataset.test_dl)

        print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(
            mean_kld_loss, mean_nll_loss))



parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--gap', type=int, default=1)
parser.add_argument('--data_path', type=str, default='C:/Aczh work place/3_paper/SNCRL_Dataset/CellCycle/')
parser.add_argument('--dataset_index', type=int, default=0)
parser.add_argument('--save_path', type=str, default='C:/Aczh work place/3_paper/SCNRL_Master/VRNN/vrnn_model_0.pkl')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--x_dim', type=int, default=44)
parser.add_argument('--h_dim', type=int, default=44 * 8)
parser.add_argument('--z_dim', type=int, default=44)
parser.add_argument('--input_size', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--n_epochs', type=int, default=25)
parser.add_argument('--clip', type=float, default=0.5)
parser.add_argument('--learning_rate', type=float, default=1e-6)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--bias', type=boolstr, default=True)
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--set_p_val', type=float, default=1e-3)


args = parser.parse_args()
args.cuda = torch.cuda.is_available() and args.gpu >= 0
device = torch.device('cuda:' + str(args.gpu) if args.cuda else 'cpu')
args.device = device
args.dataset_index=1
args.steps=1400
DS=Dataset(args.data_path,args.dataset_index)
args.input_size=int(DS.dim)
VR=vrnn(DS,args)
# manual seed
torch.manual_seed(args.seed)
# init model + optimizer + datasets
model=VRNN(args).to(device)


VR.train_vrnn(model, args)

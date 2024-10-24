import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Add the directory containing hrp.py to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import mamba
import argparse
from tqdm import tqdm

def evaluation_metric(y_test,y_hat):
    MSE = mean_squared_error(y_test, y_hat)
    RMSE = MSE**0.5
    MAE = mean_absolute_error(y_test,y_hat)
    R2 = r2_score(y_test,y_hat)
    print('%.4f %.4f %.4f %.4f' % (MSE,RMSE,MAE,R2))

def set_seed(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def dateinf(series, n_test):
    lt = len(series)
    print('Training start',series[0])
    print('Training end',series[lt-n_test-1])
    print('Testing start',series[lt-n_test])
    print('Testing end',series[lt-1])

class Net(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.config = mamba.MambaConfig(d_model=16, n_layers=2)
        self.mamba = nn.Sequential(
            nn.Linear(in_dim,16),
            mamba.Mamba(self.config),
            nn.Linear(16,out_dim)
            # nn.Tanh()
        )

    def forward(self,x):
        x = self.mamba(x)
        # out = nn.Tanh()(x)
        # prob = nn.Sigmoid()(x)
        # # Convert probability to -1 or 1 based on a 0.5 threshold
        # prediction = torch.where(prob >= 0.5, torch.tensor(1.0), torch.tensor(0.0))

        # return prediction.flatten(), prob.flatten()
        return x.flatten()

def PredictWithData(trainX, trainy, testX):
    clf = Net(len(trainX[0]),1)
    opt = torch.optim.Adam(clf.parameters(),lr=0.01,weight_decay=1e-5)
    xt = torch.from_numpy(trainX).float().unsqueeze(0)
    xv = torch.from_numpy(testX).float().unsqueeze(0)
    yt = torch.from_numpy(trainy).float()
    # if args.cuda:
    #     clf = clf.cuda()
    #     xt = xt.cuda()
    #     xv = xv.cuda()
    #     yt = yt.cuda()

    for e in tqdm(range(100), desc='Training Epochs'):
        clf.train()
        z = clf(xt)
        loss = F.mse_loss(z,yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e%10 == 0 and e!=0:
            print('Epoch %d | Lossp: %.4f' % (e, loss.item()))

    clf.eval()
    preds = clf(xv)
    # if args.cuda: mat = mat.cpu()
    yhat = preds.detach().numpy().flatten()
    # probs = probs.detach().numpy().flatten()
    return yhat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', default=False,
                        help='CUDA training.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Dimension of representations')
    parser.add_argument('--layer', type=int, default=2,
                        help='Num of layers')
    parser.add_argument('--n-test', type=int, default=300,
                        help='Size of test set')
    parser.add_argument('--ts-code', type=str, default='601988',
                        help='Stock code')

    args = parser.parse_args()
    args.cuda = args.use_cuda and torch.cuda.is_available()

    set_seed(args.seed,args.cuda)

    data = pd.read_csv(args.ts_code+'.SH.csv')
    data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
    close = data.pop('close').values
    ratechg = data['pct_chg'].apply(lambda x:0.01*x).values
    data.drop(columns=['pre_close','change','pct_chg'],inplace=True)
    dat = data.iloc[:,2:].values
    trainX, testX = dat[:-args.n_test, :], dat[-args.n_test:, :]
    trainy = ratechg[:-args.n_test]
    predictions = PredictWithData(trainX, trainy, testX)
    time = data['trade_date'][-args.n_test:]
    data1 = close[-args.n_test:]
    finalpredicted_stock_price = []
    pred = close[-args.n_test-1]
    for i in range(args.n_test):
        pred = close[-args.n_test-1+i]*(1+predictions[i])
        finalpredicted_stock_price.append(pred)

    dateinf(data['trade_date'],args.n_test)
    print('MSE RMSE MAE R2')
    evaluation_metric(data1, finalpredicted_stock_price)
    plt.figure(figsize=(10, 6))
    plt.plot(time, data1, label='Stock Price')
    plt.plot(time, finalpredicted_stock_price, label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time', fontsize=12, verticalalignment='top')
    plt.ylabel('Close', fontsize=14, horizontalalignment='center')
    plt.legend()
    plt.show()
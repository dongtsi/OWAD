import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import myutils as utils

class autoencoder(nn.Module):
    def __init__(self, feature_size):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(feature_size, int(feature_size*0.75)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.75), int(feature_size*0.5)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.5),int(feature_size*0.25)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.25),int(feature_size*0.1)))

        self.decoder = nn.Sequential(nn.Linear(int(feature_size*0.1),int(feature_size*0.25)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.25),int(feature_size*0.5)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.5),int(feature_size*0.75)),
                                     nn.ReLU(True),
                                     nn.Linear(int(feature_size*0.75),int(feature_size)),
                                     )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
getMSEvec = nn.MSELoss(reduction='none')
Params = utils.get_params('AE')
EvalParams = utils.get_params('Eval')

def se2rmse(a):
    return torch.sqrt(sum(a.t())/a.shape[1])

def train(X_train,feature_size, epoches = Params['epoches']):
    model = autoencoder(feature_size).to(device)
    optimizier = optim.Adam(model.parameters(), lr=Params['lr'], weight_decay=Params['weight_decay'])
    model.train()

    X_train = torch.from_numpy(X_train).type(torch.float)    
    if torch.cuda.is_available(): X_train = X_train.cuda()
    torch_dataset = Data.TensorDataset(X_train, X_train)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=Params['batch_size'],
        shuffle=True,
    )
    
    for epoch in range(epoches):
        for step, (batch_x, batch_y) in enumerate(loader):
            output = model(batch_x)
            loss = criterion(output, batch_y)
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()
            # if step % 10 == 0 :
        if EvalParams['verbose_info']:
            print('epoch:{}/{}'.format(epoch,epoches), '|Loss:', loss.item())
    
    model.eval()
    output = model(X_train)
    mse_vec = getMSEvec(output,X_train)
    rmse_vec = se2rmse(mse_vec).cpu().data.numpy()

    if EvalParams['verbose_info']:
        print("max AD score",max(rmse_vec))

    thres = max(rmse_vec)
    rmse_vec.sort()
    pctg = Params['percentage'] 
    thres = rmse_vec[int(len(rmse_vec)*pctg)]
    
    if EvalParams['verbose_info']:
        print("thres:",thres)

    return model, thres

@torch.no_grad()
def test(model, thres, X_test):
    model.eval()
    X_test = torch.from_numpy(X_test).type(torch.float)    
    X_test = X_test.to(device)
    output = model(X_test)
    mse_vec = getMSEvec(output,X_test)
    rmse_vec = se2rmse(mse_vec).cpu().data.numpy()
    y_pred = np.asarray([0] * len(rmse_vec))
    idx_mal = np.where(rmse_vec>thres)
    y_pred[idx_mal] = 1

    return y_pred, rmse_vec


def test_plot(X_test, rmse_vec, thres, alpha=0.15, file_name = None, label = None):
    plt.figure()
    plt.scatter(np.linspace(0,len(X_test)-1,len(X_test)),rmse_vec,s=10,alpha=alpha)
    plt.plot(np.linspace(0,len(X_test)-1,len(X_test)),[thres]*len(X_test),c='black')
    # plt.ylim(0,thres*2.)

    if label is not None:
        idx = np.where(label==1)[0]
        plt.scatter(idx,rmse_vec[idx],s=12,alpha=min(0.6, 2*alpha))
    if file_name is None:
        plt.show()
    else:
        plt.rcParams.update({'figure.dpi':300})
        plt.savefig(file_name)


    




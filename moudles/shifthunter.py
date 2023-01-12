import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, entropy
from scipy.interpolate import make_interp_spline
from sympy import Rational, symbols
from sympy.solvers.solveset import linsolve
from mlxtend.evaluate import permutation_test
import copy

import myutils as utils
try:
    DetParams = utils.get_params('ShiftDetector')
    ExpParams = utils.get_params('ShiftExplainer')
    AdaParams = utils.get_params('ShiftAdapter')
    EvalParams = utils.get_params('Eval')
except Exception as e:
    print('Error: Fail to Get Params in <configs.yml>', e)
    exit(-1)
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                
class ShiftHunter:

    def __init__(self,
                control_res, # model result in control set 
                treatment_res, # model result in treatment set 
                calibrator, # <Calibrator> Object
                bin_num = DetParams['test_bin_num'], # number of bins in the distributions 
                ad_type = 'Tab' # [Tab, Seq]
                ):
        
        assert ad_type in ['Tab', 'Seq']
        print("NOTICE: Initilize OWAD Shift Handler Under **%s** Mode!"%(ad_type))

        self.model = None
        self.ad_type = ad_type

        # calibration of initial model results
        self.calibrator = calibrator
        self.control_res = self.calibrator.process(control_res)
        self.treatment_res = self.calibrator.process(treatment_res)
        
        ## hist of two distributions
        self.bin_num = bin_num
        self.bin_array = np.linspace(0.,1.,self.bin_num+1)
        self.control_hist = np.histogram(self.control_res, self.bin_array)[0]
        self.treatment_hist = np.histogram(self.treatment_res, self.bin_array)[0]
        
    def visualize_hists(self, 
                        external = False, 
                        ##  the following params valid only if external == True  
                        res_1 = None, 
                        res_2 = None, 
                        legend_1 = None, 
                        legend_2 = None, 
                        color_1 = '#1f78b4', 
                        color_2 = '#33a02c', 
                        savefig = None, 
                        ): 
        if not external:
            res_1 = self.control_res
            res_2 = self.treatment_res
            legend_1 = 'Control Set ' # (Calibrated)
            legend_2 = 'Treatment Set ' # (Calibrated)

        cres = list(np.histogram(res_1, bins=self.bin_array))
        cres[0] = cres[0]/np.sum(cres[0])
        tres = list(np.histogram(res_2, bins=self.bin_array))
        tres[0] = tres[0]/np.sum(tres[0])

        x = (cres[1][:-1] + cres[1][1:]) / 2
        width = x[1]-x[0]
        plt.figure()
        plt.bar(x, cres[0], width=width, alpha=0.7, ec='black', label=legend_1, color=color_1)
        plt.bar(x, tres[0], width=width, alpha=0.4, ec='black', label=legend_2, color=color_2)
        
        def get_smooth_axis(res):
            x = (res[1][:-1] + res[1][1:]) / 2
            x = np.insert(x,0,0.)
            x = np.insert(x,len(x),1.)
            y = res[0]
            y = np.insert(y,0,0.)
            y = np.insert(y,len(y),0.)
            X_Y_Spline = make_interp_spline(x, y)
            X_ = np.linspace(x.min(), x.max(), 300)
            Y_ = X_Y_Spline(X_)
            return X_, Y_

        X, Y = get_smooth_axis(cres)
        plt.plot(X, Y,'-',linewidth=3, color=color_1)
        X, Y = get_smooth_axis(tres)
        plt.plot(X, Y,'-',linewidth=3, color=color_2)
        plt.ylim(ymin = 0)
        plt.legend()
        if savefig is None:
            plt.show()
        else:
            plt.savefig(savefig)

    def permu_test(self, num_rounds=DetParams['test_numrounds'], test_bin_num = None): # Permutation Test of control_hist and treatment_hist
        
        if test_bin_num is None:
            test_bin_num = self.bin_num
        def test_func(x,y): # x --> obs/treatment  y--> exp/control
            f_x = np.histogram(x, test_bin_num)[0]
            f_y = np.histogram(y, test_bin_num)[0]
            fr_E = f_y/np.sum(f_y)
            E = np.sum(f_x) * fr_E
            O = f_x
            test_stats = entropy(O, E)
            return test_stats

        p_value = permutation_test(self.treatment_res, self.control_res,
                           method='approximate',
                           num_rounds=num_rounds,
                           func = test_func,
                           seed = 42
                           )
        return p_value
    
    def explainer(self, 
                 X_t,
                 label_num,
                 lr = ExpParams['lr'], 
                 steps = ExpParams['steps'], 
                 acc_wgt = ExpParams['acc_wgt'], 
                 ohd_wgt = ExpParams['ohd_wgt'], 
                 det_wgt = ExpParams['det_wgt'], 
                 verbose = EvalParams['verbose_info'], 
                 bbin_num = ExpParams['bbin_num'],
                 discrete_thres = EvalParams['discrete_thres'],
                 plot_compose = False,
                 auto_cali_thres = True,
                ): 

        def discretize_mask(M_c, M_t, thres=discrete_thres):
            dM_c = M_c.data.cpu().numpy().copy()
            dM_c[dM_c>thres] = 1.
            dM_c[dM_c<=thres] = 0.

            dM_t = M_t.data.cpu().numpy().copy()
            dM_t[dM_t>thres] = 1.
            dM_t[dM_t<=thres] = 0.

            return dM_c, dM_t

        def visualize_explain_hist(M_c, M_t):
            dM_c, dM_t = discretize_mask(M_c, M_t)
            remove_idx_c = np.where(dM_c==1)[0]
            remain_control_res = np.delete(self.control_res,remove_idx_c)
            
            remove_idx_t=np.where(dM_t==0)[0]
            remain_treatment_res = np.delete(self.treatment_res,remove_idx_t)
            print('(Control Set) Remove:', len(remove_idx_c), 'Remain:', len(remain_control_res))
            print('(Treatment Set) Remove:', len(remove_idx_t), 'Remain:', len(remain_treatment_res))

            explain_res = np.concatenate((remain_control_res,remain_treatment_res))

            self.visualize_hists(external=True,  
                        res_1 = explain_res, 
                        res_2 = self.treatment_res,  
                        legend_1 = 'Explain Set',
                        legend_2 = 'Treatment Set', color_1='#fb9a99')

        def visualize_compose_results(M_c, M_t):
            dM_c, dM_t = discretize_mask(M_c, M_t)
            remove_idx_c = np.where(dM_c==1)[0]
            remain_control_res = np.delete(self.control_res,remove_idx_c)
            
            remove_idx_t=np.where(dM_t==0)[0]
            remain_treatment_res = np.delete(self.treatment_res,remove_idx_t)

            cres = list(np.histogram(remain_control_res, bins=self.bin_array))
            cres[0] = cres[0]
            tres = list(np.histogram(remain_treatment_res, bins=self.bin_array))
            tres[0] = tres[0]

            x = (cres[1][:-1] + cres[1][1:]) / 2
            width = x[1]-x[0]
            plt.figure()
            plt.bar(x, cres[0], width=width, alpha=0.7, ec='black', label='control set (remain)', color='#1f78b4')
            plt.bar(x, tres[0], width=width, bottom=cres[0], alpha=0.7, ec='black', label='treatment set (remain)', color='#33a02c')
            plt.legend()

        def get_normal_t_idx(pctg=0.01): # percentage for threshold selection
            # min_c, max_c = np.min(self.control_res), np.max(self.control_res)
            c_res_tmp = np.sort(self.control_res)
            min_c = c_res_tmp[int(len(c_res_tmp)*pctg)]
            max_c = c_res_tmp[int(len(c_res_tmp)*(1-pctg))]
            normal_t_idx = np.where((self.treatment_res>=min_c)&(self.treatment_res<=max_c))[0]
            return normal_t_idx
        
        def get_abnormal_t_idx(pctg=0.01):
            # min_c, max_c = np.min(self.control_res), np.max(self.control_res)
            c_res_tmp = np.sort(self.control_res)
            min_c = c_res_tmp[int(len(c_res_tmp)*pctg)]
            max_c = c_res_tmp[int(len(c_res_tmp)*(1-pctg))]
            idx_1 = np.where(self.treatment_res<min_c)[0]
            idx_2 = np.where(self.treatment_res>max_c)[0]
            abnormal_t_idx = np.concatenate((idx_1,idx_2))
            return abnormal_t_idx
        
        control_res = torch.from_numpy(self.control_res).type(torch.float).to(device)
        treatment_res = torch.from_numpy(self.treatment_res).type(torch.float).to(device)

        M_c = torch.empty_like(control_res) 
        M_t = torch.empty_like(treatment_res)
        
        M_c = torch.nn.init.uniform_(M_c, a=0., b=1.)
        M_t = torch.ones_like(M_t)
        M_t[get_normal_t_idx()] = torch.nn.init.uniform_(M_t[get_normal_t_idx()], a=0., b=1.)

        M_c.requires_grad = True
        M_t.requires_grad = True

        # optimizer = optim.Adam([M_c, M_t], lr=lr)
        optimizer = optim.SGD([M_c, M_t], lr=lr)
        delta = 1e-4

        for step in range(steps):
            
            ## clipping to (0, 1)
            with torch.no_grad():
                M_c[:] = torch.clamp(M_c, delta, 1-delta)
                M_t[:] = torch.clamp(M_t, delta, 1-delta)

            ## Accuracy_Loss: discrete distribution (bin-wise, L2)
            Accuracy_Loss = 0.
            bin_array = np.linspace(0.,1.,bbin_num+1)
            for i in range(bbin_num):
                mmask_c = torch.zeros_like(control_res)
                mmask_c[(control_res>bin_array[i])&(control_res<bin_array[i+1])] = 1.
                mmask_t = torch.zeros_like(treatment_res[get_normal_t_idx()])
                mmask_t[(treatment_res[get_normal_t_idx()]>bin_array[i])&(treatment_res[get_normal_t_idx()]<bin_array[i+1])] = 1.
                mmask_t_ab = torch.zeros_like(treatment_res[get_abnormal_t_idx()])
                mmask_t_ab[(treatment_res[get_abnormal_t_idx()]>bin_array[i])&(treatment_res[get_abnormal_t_idx()]<bin_array[i+1])] = 1.
                bin_obs = (torch.sum((1-M_c)*mmask_c)+torch.sum(M_t[get_normal_t_idx()]*mmask_t)+ torch.sum(mmask_t_ab))\
                          /(torch.sum(1-M_c)+torch.sum(M_t[get_normal_t_idx()])+len(mmask_t_ab))
                treatment_hist = np.histogram(self.treatment_res, bin_array)[0]
                bin_tgt = (treatment_hist/len(self.treatment_res))[i]
                bin_loss = torch.norm(bin_obs-bin_tgt, p=2)
                Accuracy_Loss += bin_loss
        

            Overhead_Loss =  torch.norm(M_c, p=2) + torch.norm(M_t[get_normal_t_idx()], p=2)
            Determinism_Loss = -(torch.sum(M_c * torch.log(M_c) + (1-M_c) * torch.log(1-M_c)) 
                               + torch.sum(M_t[get_normal_t_idx()] * torch.log(M_t[get_normal_t_idx()]) + (1-M_t[get_normal_t_idx()]) * torch.log(1-M_t[get_normal_t_idx()])))

            if step == 0:
                cali_acc = 1/Accuracy_Loss.item()
                cali_ohd = 1/Overhead_Loss.item()
                cali_det = 1/Determinism_Loss.item()
                
            if auto_cali_thres:
                Loss =  cali_acc*acc_wgt*Accuracy_Loss + cali_ohd*ohd_wgt * Overhead_Loss+ cali_det*det_wgt * Determinism_Loss
                if step % 10 == 0 or step == steps-1:
                    if verbose:
                        mc = M_c.data.cpu().numpy()
                        mt = M_t.data.cpu().numpy()
                        print('step:{}'.format(step), '|Loss:%.4f'%Loss.item(), 
                                '|Accuracy_Loss:%.4f'%(cali_acc*Accuracy_Loss.item()),
                                '|Overhead_Loss:%.4f'%(cali_ohd*Overhead_Loss.item()),
                                '|Determinism_Loss:%.4f'%(cali_det*Determinism_Loss.item()),
                                # '|M_c Sum:%d'%int(torch.sum(M_c).item()),
                                # '|M_t Sum:%d'%int(torch.sum(M_t).item()),
                                '|M_c num:', len(mc[mc>EvalParams['discrete_thres']]), 
                                '|M_t num:', len(mt[mt>EvalParams['discrete_thres']])         
                        )
                        
            else:
                Loss =  acc_wgt * Accuracy_Loss + ohd_wgt * Overhead_Loss+ det_wgt * Determinism_Loss
                if step % 10 == 0 or step == steps-1:
                    if verbose:
                        print(Accuracy_Loss.item()*acc_wgt, Overhead_Loss.item()*ohd_wgt, Determinism_Loss.item()*det_wgt)
                        mc = M_c.data.cpu().numpy()
                        mt = M_t.data.cpu().numpy()
                        print('step:{}'.format(step), '|Loss:%.4f'%Loss.item(), 
                                '|Accuracy_Loss:%.4f'%Accuracy_Loss.item(),
                                '|Overhead_Loss:%.4f'%Overhead_Loss.item(),
                                '|Determinism_Loss:%.4f'%Determinism_Loss.item(),
                                # '|M_c Sum:%d'%int(torch.sum(M_c).item()), 
                                # '|M_t Sum:%d'%int(torch.sum(M_t).item()), 
                                '|M_c num:', len(mc[mc>EvalParams['discrete_thres']]), 
                                '|M_t num:', len(mt[mt>EvalParams['discrete_thres']])           
                        )
                        
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()


        if plot_compose: 
            visualize_explain_hist(M_c, M_t)
            visualize_compose_results(M_c, M_t)

        self.M_c = M_c
        self.M_t = M_t

        self.X_tre = X_t
        _, remain_X_tre, remain_X_con = self.get_remain_X(self.calibrator.X_con, self.X_tre, thres = discrete_thres, label_num = label_num)
        delete_X_con = self.get_del_Xc(self.calibrator.X_con)

        self.EXPLAIN_RES = {
            'remain_X_tre': remain_X_tre,
            'delete_X_con': delete_X_con,
            'remain_X_con': remain_X_con,
        }
        return self.EXPLAIN_RES

    def adapter(self, model, **kvargs):
        print('Adapter: Begin Processing ... ') 
        if self.ad_type == 'Tab':
            self._adapter_tab(model, **kvargs)
        elif self.ad_type == 'Seq': 
            self._adapter_seq(model, **kvargs)
       
    def _adapter_tab(self,
            model,
            param_weight = None, 
            lr = AdaParams['lr'],  
            reg_wgt = AdaParams['reg_wgt'],  
            steps = AdaParams['steps'], 
            batch_size = AdaParams['batch_size'], 
            fast_wgt_est = 5, 
            init_thres = 1-1e5, 
            verbose = EvalParams['verbose_info'], 
            ): 

        self.model = copy.deepcopy(model)

        def np2ts(X):
            return torch.from_numpy(X).type(torch.float).to(device)
 
        def get_params_weight(X,  
                              M): 
            X = np2ts(X)
            Weight = []
            for i in range(len(X)):
                loss = torch.norm(self.model(X[i:i+1]), p=2)
                loss.backward()
                W = []
                for p in self.model.parameters():
                    W.append(p.grad.clone())
                for j in range(len(W)):
                    W[j] = W[j] * M[j]
                    if len(Weight)<len(W):
                        Weight.append(W[j])
                    else:
                        Weight[j] += W[j]
                if i%2000==0:
                    print(' Estimating Params Weight:%d/%d'%(i,len(X)))
            for j in range(len(Weight)):
                Weight[j] /= len(X)
            return Weight

        def get_params_list():
            params_list = []
            for p in self.model.parameters():
                if p.requires_grad:
                    params_list.append(p.clone())
            return params_list

        def se2rmse(a):
            return torch.sqrt(sum(a.t())/a.shape[1])
 
        self.model.train()
        getMSEvec = nn.MSELoss(reduction='none')
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        init_params = get_params_list()

        remain_X = np.concatenate((self.EXPLAIN_RES['remain_X_con'], self.X_tre),axis=0)
        self.calibrator.X_con = remain_X
        remain_X = np2ts(remain_X)

        if param_weight is None:
            X = self.calibrator.X_con
            M = (1-self.M_c).data.cpu().numpy()
            self.param_weight = get_params_weight(X[::fast_wgt_est], M[::fast_wgt_est])
        else:
            self.param_weight = param_weight

        torch_dataset = Data.TensorDataset(remain_X, remain_X)
        loader = Data.DataLoader(
                dataset=torch_dataset,
                batch_size=batch_size,
                shuffle=True,
            )

        for epoch in range(steps):
            for step, (batch_X, _) in enumerate(loader):
            
                Distrib_Loss = criterion(self.model(batch_X),batch_X)
                Regularization_Loss = torch.tensor(0.).to(device)
                
                for i,p in enumerate(self.model.parameters()):
                    if p.requires_grad:
                        Regularization_Loss += torch.norm((p-init_params[i]*init_thres)*self.param_weight[i])

                Loss = Distrib_Loss/Distrib_Loss.detach() + reg_wgt * Regularization_Loss/Regularization_Loss.detach()


                if step % 50 == 0: # and step:
                    if verbose:
                        print('epoch:{}'.format(epoch), ',step:{}'.format(step),
                            #   '|Loss:%.4f'%Loss.item(),
                                '|Distrib_Loss:%.4f'%(Distrib_Loss).item(),
                                'rmse_vec:%.4f'%torch.mean(se2rmse(getMSEvec(self.model(remain_X), remain_X))).item(),
                                # '|Regularization_Loss:%.4f'%(reg_wgt * Regularization_Loss).item()
                        )                    
                optimizer.zero_grad()
                Loss.backward()
                optimizer.step()

    def _adapter_seq(self,
                model,
                param_weight = None, 
                lr = AdaParams['lr'],  
                reg_wgt = AdaParams['reg_wgt'],  
                steps = AdaParams['steps'], 
                batch_size = AdaParams['batch_size'], 
                fast_wgt_est = 5, 
                init_thres = 1-1e5, 
                verbose = EvalParams['verbose_info'], 
                ): 

        self.model = copy.deepcopy(model)
        def get_params_weight(X,  
                              M): 

            X = torch.tensor(X, dtype=torch.long)
            X = X.clone().detach().view(-1, utils.get_params('DeepLog')['window_size']).to(device)
            X = F.one_hot(X,num_classes=utils.get_params('DeepLog')['num_classes']).float()
            
            Weight = []
            for i in range(len(X)):
                loss = torch.norm(self.model(X[i:i+1]), p=2)
                loss.backward()
                W = []
                for p in self.model.parameters():
                    W.append(p.grad.clone())
                for j in range(len(W)):
                    W[j] = W[j] * M[j]
                    if len(Weight)<len(W):
                        Weight.append(W[j])
                    else:
                        Weight[j] += W[j]
                if i%1000==0:
                    print('Estimating Params Weight:%d/%d'%(i,len(X)))
            for j in range(len(Weight)):
                Weight[j] /= len(X)
            return Weight

        def get_params_list():
            params_list = []
            for p in self.model.parameters():
                if p.requires_grad:
                    params_list.append(p.clone())
            return params_list
 
        self.model.train()

        remain_X = {
            'input':    np.concatenate((self.EXPLAIN_RES['remain_X_con']['input'],self.X_tre['input']),axis=0),
            'output':   np.concatenate((self.EXPLAIN_RES['remain_X_con']['output'],self.X_tre['output']),axis=0)
        }

        
        init_params = get_params_list()
        self.calibrator.X_con = remain_X

        if param_weight is None:
            X = self.calibrator.X_con['input']
            M = (1-self.M_c).data.cpu().numpy()
            self.param_weight = get_params_weight(X[::fast_wgt_est], M[::fast_wgt_est])
        else:
            self.param_weight = param_weight


        X_input, X_output = remain_X['input'], remain_X['output']
        dataset = Data.TensorDataset(torch.tensor(X_input, dtype=torch.long), torch.tensor(X_output))
        dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),lr=lr)
        
        for epoch in range(steps): 
            for step, (seq, label) in enumerate(dataloader):
                seq = seq.clone().detach().view(-1, utils.get_params('DeepLog')['window_size']).to(device)
                seq = F.one_hot(seq,num_classes=utils.get_params('DeepLog')['num_classes']).float()
                output = model(seq)
                Distrib_Loss = criterion(output, label.to(device))

                Regularization_Loss = torch.tensor(0.).to(device)
                for i,p in enumerate(self.model.parameters()):
                    if p.requires_grad:
                        Regularization_Loss += torch.norm((p-init_params[i]*init_thres)*self.param_weight[i])

                Loss = Distrib_Loss/Distrib_Loss.detach() + reg_wgt * Regularization_Loss/Regularization_Loss.detach()

                if step % 50 == 0: 
                    if verbose:
                        output_label = label.detach().numpy()
                        print('epoch:{}'.format(epoch), ',step:{}'.format(step), 
                                #    '|Loss:%.4f'%Loss.item(),
                                '|Distrib_Loss:%.4f'%(Distrib_Loss).item(),
                                '|benign probs:', np.mean(output.cpu().detach().numpy()[np.arange(len(output_label)),output_label]),
                                # '|Regularization_Loss:%.4f'%(reg_wgt * Regularization_Loss).item()
                        )

                ## Backward and optimize
                optimizer.zero_grad()
                Loss.backward()
                optimizer.step()

    def get_remain_X(self, 
                X_c, X_t, 
                thres = EvalParams['discrete_thres'], 
                label_num = None,
                retain_Xnum = False, 
            ):

        M_c = self.M_c.data.cpu().numpy().copy()
        M_c[M_c>thres] = 1.
        M_c[M_c<=thres] = 0.

        M_t = self.M_t.data.cpu().numpy().copy()
        M_t[M_t>thres] = 1.
        M_t[M_t<=thres] = 0.

        if self.ad_type == 'Seq':

            print('get X_c, X_t (len):', len(X_c['input']), len(X_t['input']))

            remain_X_c = {'input':X_c['input'][M_c==0], 'output': X_c['output'][M_c==0]}
            remain_X_t = {'input':X_t['input'][M_t==1], 'output': X_t['output'][M_t==1]}

            if label_num is not None:
                if label_num>=len(X_t['input']) or label_num>=len(remain_X_t['input']):
                    print("** Warning: No Need to Clip <remain_X_t> according to <label_num>")
                    print("   (label_num:%d, remain_X_t:%d, X_t:%d)"%(label_num, len(remain_X_t['input']), len(X_t['input'])))
                else:
                    print("** Cliping <remain_X_t> ...")
                    M_t = self.M_t.data.cpu().numpy().copy()
                    sortidx = np.argsort(-M_t)
                    remain_X_t = {'input':X_t['input'][sortidx[:label_num]], 'output': X_t['output'][sortidx[:label_num]]}

                
                if retain_Xnum:
                    if label_num + len(self.calibrator.X_con['input']) <= EvalParams['control_num']:
                        print("** Warning: No Need to Clip <remain_X_c> according to <label_num>")
                        print("   (label_num:%d, X_con:%d, Retain_Num:%d)"%(label_num, len(self.calibrator.X_con['input']), EvalParams['control_num']))
                    else:
                        print("** Cliping <remain_X_c> ...")
                        M_c = self.M_c.data.cpu().numpy().copy()
                        sortidx = np.argsort(M_c)
                        remain_X_c = {'input':  X_c['input'][sortidx[:(EvalParams['control_num']-label_num)]], 
                                        'output': X_c['output'][sortidx[:(EvalParams['control_num']-label_num)]]}

            print('Remain X_c, X_t (len):', len(remain_X_c['input']), len(remain_X_t['input']))

            remain_X = {
                'input':    np.concatenate((remain_X_c['input'],remain_X_t['input']),axis=0),
                'output':   np.concatenate((remain_X_c['output'],remain_X_t['output']),axis=0)
            }

        elif self.ad_type == 'Tab':
            
            remain_X_c = X_c[M_c==0]
            remain_X_t = X_t[M_t==1]

            if label_num is not None:
                if label_num>=len(X_t) or label_num>=len(remain_X_t):
                    print("** Warning: No Need to Clip <remain_X_t> according to <label_num>")
                    print("   (label_num:%d, remain_X_t:%d, X_t:%d)"%(label_num, len(remain_X_t), len(X_t)))
                else:
                    print("** Cliping <remain_X_t> ...")
                    M_t = self.M_t.data.cpu().numpy().copy()
                    sortidx = np.argsort(-M_t)
                    remain_X_t = X_t[sortidx[:label_num]]

                if retain_Xnum: 
                    if label_num + len(self.calibrator.X_con) <= EvalParams['control_num']:
                        print("** Warning: No Need to Clip <remain_X_c> according to <label_num>")
                        print("   (label_num:%d, X_con:%d, Retain_Num:%d)"%(label_num, len(self.calibrator.X_con), EvalParams['control_num']))
                    else:
                        print("** Cliping <remain_X_c> ...")
                        M_c = self.M_c.data.cpu().numpy().copy()
                        sortidx = np.argsort(M_c)
                        remain_X_c = X_c[sortidx[:(len(self.calibrator.X_con)-label_num)]]                    
            
            print('Remain X_c.shape', remain_X_c.shape, 'X_t.shape', remain_X_t.shape)
            remain_X = np.concatenate((remain_X_c,remain_X_t),axis=0)

        return remain_X, remain_X_c, remain_X_t

    def get_del_Xc(self,
                    X_c, 
                    thres = EvalParams['discrete_thres'],
        ):
        M_c = self.M_c.data.cpu().numpy().copy()
        M_c[M_c>thres] = 1.
        M_c[M_c<=thres] = 0.

        if self.ad_type == 'Tab':
            del_Xc = X_c[M_c==1]
        elif self.ad_type == 'Seq':
            del_Xc = {'input':X_c['input'][M_c==1], 'output': X_c['output'][M_c==1]}
        return del_Xc

    def set_admodel(self, model):
        self.model = model 

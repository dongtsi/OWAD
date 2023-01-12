import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.isotonic import IsotonicRegression
import myutils as utils

class Calibrator:
    def __init__(self, 
                X_con,
                method = 'Isotonic',  
                ):

        assert method in ['MLS', 'PWLF', 'Isotonic', 'Uncalibrate']
        print("NOTICE: Initialize OWAD Calibrator Under **%s** Method!"%(method))

        self.X_con = X_con
        self.method = method
        
        if self.method == 'MLS':
            self.A, self.B = 60. , 0 # paramters for 'MLS' method
        elif self.method == 'PWLF': 
            self.Breaks = dict() # paramters for 'PWLF' method
        elif self.method == 'Isotonic': 
            self.cali_model = None
        elif self.method == 'Uncalibrate':
            pass
        else:
            print('Error Params <method>')
            exit(-1)
    

    def process(self,   
                model_res):
        
        if self.method == 'Uncalibrate': # return Uncalibrated model outputs
            return model_res
        
        if self.method == 'MLS':
            return 2/(1 + np.exp(self.A*(model_res+self.B)))

        elif self.method == 'PWLF':
            x_breaks, y_breaks = self.Breaks['x'], self.Breaks['y']
            k_list = []
            b_list = []
            for i in range(len(x_breaks)-1):
                x_coords = (x_breaks[i],x_breaks[i+1])
                y_coords = (y_breaks[i],y_breaks[i+1])
                A = np.vstack([x_coords,np.ones(len(x_coords))]).T
                k, b = np.linalg.lstsq(A, y_coords)[0]
                k_list.append(k)
                b_list.append(b)
            
            def pwl_func(x):
                
                if y_breaks[-1] - y_breaks[0] > 0:
                    direction = 1. # inc 

                else:
                    direction = 0. # dec

                if x <= x_breaks[0]:
                    y = 1. - direction
                    return y
                if x >= x_breaks[len(x_breaks)-1]:
                    y = direction
                    return y
                for i in range(len(x_breaks)-1):
                    if x_breaks[i] <= x < x_breaks[i+1]:
                        y = k_list[i]*x + b_list[i]
                        y = min(y, 1.)
                        y = max(y, 0.)
                        return y
            
            calibrated_res = []
            for res in model_res:
                calibrated_res.append(pwl_func(res))
            return np.asarray(calibrated_res)

        elif self.method == 'Isotonic':
            return self.cali_model.predict(model_res)
                
    def set_calibrator(self, 
                       P_con,
                       is_P_mal = True,
                       ):         
        
        if is_P_mal:
            print('NOTICE: uncalibrated Probs is **MALICIOUS** confidence')
            index_sort = np.argsort(-P_con)
        else:
            print('NOTICE: uncalibrated Probs is **BENIGN** confidence')
            index_sort = np.argsort(P_con)

        x_group = P_con
        y_group = []
        for i in range(len(P_con)):
            y_group.append(np.where(index_sort==i)[0][0]/len(P_con))
        y_group = np.asarray(y_group)
        
        if self.method == 'MLS':
            def func(x, A, B):
                return 2/(1 + np.exp(A*(x+B)))    
            self.A, self.B = curve_fit(func, x_group, y_group, maxfev=10000)[0]

        
        elif self.method == 'PWLF':
            import pwlf
            my_pwlf = pwlf.PiecewiseLinFit(x_group, y_group, degree=1)
            try:
                break_num = utils.get_params('PWLF_break_num')
            except Exception as e:
                print('Error: Fail to Get Params in <configs.yml>', e)
                exit(-1)
            x_breaks = my_pwlf.fit(break_num)
            y_breaks =  my_pwlf.predict(x_breaks)
            self.Breaks['x'], self.Breaks['y'] = x_breaks, y_breaks

        elif self.method == 'Isotonic':
            sort_idx = np.argsort(x_group)
            x_group = x_group[sort_idx]
            y_group = y_group[sort_idx]
            
            self.cali_model = IsotonicRegression(y_min=0., y_max=1., increasing=(not is_P_mal), out_of_bounds='clip').fit(x_group, y_group)

        
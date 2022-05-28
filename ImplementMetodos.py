import numpy as np
import pandas as pd
#mover a carpeta Estadística II
#Implementación de ecuación normal
def beta(x_train,y_train):
    n_data=x_train.shape[0]
    ones=np.ones((n_data,1))
    A=np.append(ones,x_train,axis=1)
    A_t=A.T
    beta=(np.linalg.inv(A_t@A)@A_t)@y_train
    return beta


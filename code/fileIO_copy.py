import pandas as pd
import numpy as np
from scipy.stats import skew
from random import randint
from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE
from imblearn.over_sampling import ADASYN # doctest: +NORMALIZE_WHITESPACE
from imblearn.under_sampling import CondensedNearestNeighbour # doctest: +SKIP



def readFile(path, y_label,method, encode_features=[], skew_exempted=[], training_ratio=0.7, shuffle=True, needSkew=False,fea_eng=True):
    raw = pd.read_csv(path)
    n, d = raw.shape
   

    if (shuffle):
        raw = raw.sample(frac=1).reset_index(drop=True)  # shuffle
    
    if (needSkew):
        skewed = raw[raw.dtypes[raw.dtypes != "object"].index.drop(skew_exempted)].apply(lambda x: skew(x.dropna()))
        skewed = skewed[skewed > 0.75].index
        raw[skewed] = np.log1p(raw[skewed])  # reduce skewness
    
    raw = pd.get_dummies(raw, columns=encode_features)  # encode categorical features
    raw = raw.fillna(raw.mean())
    # if(method=='OverSample'):
    #     ind_more=np.argmax(np.bincount(raw[y_label]))
    #     more=raw[ind]
    #     less=raw[-ind]
    #     x = [randint(0, len(less)) for a in range(0, len(more)-len(less))]
    #     raw.
    X=raw.drop(y_label,axis=1)
    y=raw[y_label]
    if(method=='OverSample'):        
        ada = ADASYN(random_state=42)
        X_res, y_res = ada.fit_resample(X, y)
        X=X_res
        y=y_res
    if(method=='UnderSample'): 
        # for i in []   
        model = CondensedNearestNeighbour(random_state=42) # doctest: +SKIP
        X_res, y_res = model.fit_resample(X, y) #doctest: +SKIP    \      
        X=X_res
        y=y_res
    # if(method=='Weights'): 
    # if(fea_eng==True):
    #     # X,y=feature_eng(X,y)
    X_train, X_test, y_train, y_test=split(X,y, training_ratio)
    return X_train, X_test, y_train, y_test
   


def split(X,y, training_ratio=0.7):
    n, d = X.shape
    training_size = int(n * training_ratio)
    X_train = X[0:training_size]
    X_test = X[training_size:]
    y_train = y[0:training_size]
    y_test =y[training_size:]
    return X_train, X_test, y_train, y_test

# def feature_eng(X,y):

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import os
import gc
from sklearn.base import BaseEstimator, ClassifierMixin
from joblib import load, dump
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score, accuracy_score
import sys
sys.path.append('/raid/lzyt_dir/Test/MALDINetTest/MALDIMap')
from net import _MALDINet

class MultiClassEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self,
               epochs = 60,
               kernel_block = 'MK',
               units = [12, 24, 48],
               dense_layers = [256, 128, 64, 32],
               dense_avf = 'relu',
               batch_size = 128,
               lr = 1e-4,
               loss = 'categorical_crossentropy',
               n_inception = 2,
               dropout = 0.0,
               monitor = 'val_loss',
               metric = 'ACC',
               early_stop = False,
               patience = 10000,
               verbose = 0, 
               last_avf = 'softmax',
               random_state = 32,
               gpuid=0,
               name = 'MALDINet MultiClass Estimator',
               printOrNot = False):
        
        self.epochs = epochs
        self.kernel_block = kernel_block
        self.units = units
        self.dense_layers = dense_layers
        self.dense_avf = dense_avf
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        self.last_avf = last_avf
        
        self.n_inception = n_inception      
        self.dropout = dropout
        
        self.monitor = monitor
        self.metric = metric
        self.early_stop = early_stop
        self.patience = patience
        
        self.gpuid = str(gpuid)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpuid
        os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
        physical_gpus = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_gpus[0], True)    #动态调用GPU

        self.verbose = verbose
        self.random_state = random_state
        self.name = name
        if (printOrNot): print(self)
        
    def get_params(self, deep = True):
        model_paras =  {"epochs": self.epochs,
                        "lr":self.lr, 
                        "loss":self.loss,
                        "kernel_block": self.kernel_block,
                        "units":self.units,
                        "dense_layers": self.dense_layers,
                        "dense_avf":self.dense_avf,
                        "last_avf":self.last_avf,
                        "batch_size":self.batch_size, 
                        "dropout":self.dropout,
                        "n_inception":self.n_inception,                     
                        "monitor": self.monitor,
                        "metric":self.metric,
                        "early_stop":self.early_stop,
                        "patience":self.patience,
                        "random_state":self.random_state,
                        "verbose":self.verbose,
                        "name":self.name,
                        "gpuid": self.gpuid,
                       }
        return model_paras
    
    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, y, X_val = None, y_val = None, class_weights = None):
        # if (X_val is None) | (y_val is None):
        #     X_val = X
        #     y_val = y
        np.random.seed(self.random_state)
        tf.compat.v1.set_random_seed(self.random_state)

        model = _MALDINet(input_shape = X.shape[1:],
                          n_outputs = y.shape[-1],
                          kernel_block = self.kernel_block,
                          units = self.units,
                          dense_layers = self.dense_layers,
                          n_inception = self.n_inception,
                          dense_avf = self.dense_avf,
                          dropout = self.dropout,
                          last_avf = self.last_avf)
        opt = tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #
        if (self.metric == 'ACC'): model.compile(optimizer = opt, loss = self.loss, metrics = ['accuracy'])
        if (self.metric == 'AUC'): model.compile(optimizer = opt, loss = self.loss,  metrics=[tf.metrics.AUC(multi_label=False)])

        # checkpoint = ModelCheckpoint('weights/epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5',
        #                      monitor=self.monitor,
        #                      save_best_only=True,
        #                      mode='auto')
        if (X_val is None) | (y_val is None):
            history = model.fit(X,y,
                            batch_size=self.batch_size,
                            epochs= self.epochs,
                            verbose= self.verbose,
                            shuffle = True,
                            class_weight = class_weights,
                            )
        else: 
            early_stopping = EarlyStopping(monitor=self.monitor, patience=self.patience, restore_best_weights=True)
            history = model.fit(X,y,
                            batch_size=self.batch_size,
                            epochs= self.epochs,
                            verbose= self.verbose,
                            shuffle = True,
                            class_weight = class_weights,
                            validation_data = (X_val, y_val),
                            callbacks=[early_stopping] if self.early_stop else None,
                            )
        
        self._model = model
        self.history = history
        return self
    
    def predict(self, X):
        probs = self._model.predict(X)
        y_pred = np.argmax(probs, axis=1)
        return y_pred
    
    def predict_proba(self, X):
        probs = self._model.predict(X)
        return probs
    
    def score(self, X, y, scoring = 'acc'):
        y_pred = self.predict(X)
        if (scoring == 'auc'): 
            score = roc_auc_score(y[:,1], y_pred)
        else: 
            score = accuracy_score(y[:,1], y_pred)
        return score
    
    def clean(self):
        del self.model
        del self
        gc.collect()
        K.clear_session()

class MultiLabelEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self,
               epochs = 60,
               kernel_block = 'MK',
               units = [12, 24, 48],
               dense_layers = [256, 128, 64, 32],
               dense_avf = 'relu',
               last_avf = 'sigmoid',
               batch_size = 128,
               lr = 1e-4,
               loss = tf.nn.sigmoid_cross_entropy_with_logits,
               n_inception = 2,
               dropout = 0.0,
               monitor = 'val_loss',
               metric = 'ROC',
               patience = 10000,
               verbose = 0, 
               random_state = 32,
               gpuid=0,
               name = 'MALDINet MultiLabel Estimator',
               printOrNot = False):
        
        self.epochs = epochs
        self.kernel_block = kernel_block
        self.units = units
        self.dense_layers = dense_layers
        self.dense_avf = dense_avf
        self.last_avf = last_avf
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        
        self.n_inception = n_inception      
        self.dropout = dropout
        
        self.monitor = monitor
        self.metric = metric
        self.patience = patience
        
        self.gpuid = str(gpuid)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpuid
        os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
        physical_gpus = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_gpus[0], True)    #动态调用GPU

        self.verbose = verbose
        self.random_state = random_state
        self.name = name
        if (printOrNot): print(self)
        
    def get_params(self, deep = True):
        model_paras =  {"epochs": self.epochs,
                        "lr":self.lr, 
                        "loss":self.loss,
                        "kernel_block": self.kernel_block,
                        "units":self.units,
                        "dense_layers": self.dense_layers,
                        "dense_avf":self.dense_avf,
                        "last_avf":self.last_avf,
                        "batch_size":self.batch_size, 
                        "dropout":self.dropout,
                        "n_inception":self.n_inception,                     
                        "monitor": self.monitor,
                        "metric":self.metric,
                        "patience":self.patience,
                        "random_state":self.random_state,
                        "verbose":self.verbose,
                        "name":self.name,
                        "gpuid": self.gpuid,
                       }
        return model_paras
    
    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, y, X_val = None, y_val = None):
        # if (X_val is None) | (y_val is None):
        #     X_val = X
        #     y_val = y
        np.random.seed(self.random_state)
        tf.compat.v1.set_random_seed(self.random_state)

        model = _MALDINet(input_shape = X.shape[1:],
                          n_outputs = y.shape[-1],
                          kernel_block = self.kernel_block,
                          units = self.units,
                          dense_layers = self.dense_layers,
                          n_inception = self.n_inception,
                          dense_avf = self.dense_avf,
                          dropout = self.dropout,
                          last_avf = self.last_avf)
        opt = tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #
        model.compile(optimizer = opt, loss = self.loss,  metrics=[tf.metrics.AUC(multi_label=True)])

        # checkpoint = ModelCheckpoint('weights/epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5',
        #                      monitor=self.monitor,
        #                      save_best_only=True,
        #                      mode='auto')
        if (X_val is None) | (y_val is None):
            history = model.fit(X,y,
                            batch_size=self.batch_size,
                            epochs= self.epochs,
                            verbose= self.verbose,
                            shuffle = True,
                            #validation_data = (X_val, y_val),
                            #callbacks=[checkpoint]
                            )
        else: history = model.fit(X,y,
                            batch_size=self.batch_size,
                            epochs= self.epochs,
                            verbose= self.verbose,
                            shuffle = True,
                            validation_data = (X_val, y_val),
                            #callbacks=[checkpoint]
                            )
        
        self.model = model
        self.history = history
        return self
    
    def predict(self, X):
        probs = self.predict_proba(X)
        y_pred = np.round(probs)
        return y_pred
    
    def predict_proba(self, X):
        probs = self.model.predict(X)
        return probs
    
    def clean(self):
        del self.model
        del self
        gc.collect()
        K.clear_session()
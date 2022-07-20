import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import StandardScaler,scale
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.inspection import permutation_importance

import itertools
import random

class All_in_one:
    
    def __init__(self,case:str,model_name:str,output:str,test_size:int):
        self.case = case.lower()
        self.model = model_name.lower()
        self.output = output.lower()
        self.test_size = test_size
        if self.case  not in ['c1','c2']:
            sys.exit('For case 1, write C1 or c1 and case 2 write C2 or c2')
            
        if self.model  not in ['lr','knn','svr','pca','sgdr','plsr1','plsr2']:
            sys.exit('lr,knn,svr,pca,sgdr,plsr1','plsr2')
            
        if self.output  not in ['temperature','heat','efficiency','all']:
            sys.exit('temperature,heat,efficiency,all')
        
        if self.model == 'plsr2' and self.output!='all':
            sys.exit("Please pass in output as all for Partial Least Square Regression")
            
    def read_data(self):
        #print("Reading data")
        '''
        Reading the data from csv file based on the case
        '''
        random.seed(42)
        list_ = random.sample(range(0, 35),self.test_size)
        df_test,df_train = pd.DataFrame(),pd.DataFrame()
        try:
            if self.case  == "c1":
                df_data = pd.read_csv("CASE_1.csv")
            if self.case  == "c2":
                df_data = pd.read_csv("CASE_2.csv")
        except:
            print("The data frame name is incorrect while passing through read_data")
        for i in list_:
            df_test = df_test.append(df_data.loc[i])
        
        df_test = df_test.reindex(columns = df_data.columns)
        df_test = df_test.reset_index(drop=True)
        df_train = df_data.drop(list_,axis=0)
        df_train = df_train.reset_index(drop=True)
        return df_data,df_train,df_test
        

    def data_extract(self,df):
        '''
        Getting the X and Y columns
        y_t is for temperature
        y_e is for efficiency
        y_q is for heat
        '''
        X = df[df.columns[[1,2,3,4]]]
        y_t = df[df.columns[[5]]]
        y_q = df[df.columns[[6]]]
        y_e = df[df.columns[[7]]]
        y = df[df.columns[[5,6,7]]]
        return  X,y,y_t,y_q,y_e 
    
    def pre_processing(self,train:bool):
        #print("Pre processing data")

        df_data,df_train,df_test = self.read_data()
        
        if train:
             X,y,y_t,y_q,y_e = self.data_extract(df_train)
        else:
             X,y,y_t,y_q,y_e = self.data_extract(df_test)
        return X,y,y_t,y_q,y_e
    
    
    def models(self,num_neighbours=1,comp=1,kernel="linear"):
        #print("Choosing model")
        '''
        Getting the model ready
        '''
        try:
            if self.model  == "lr":
                model = Pipeline([('scl', StandardScaler()),
                              ('lr',LinearRegression())])
                
            elif self.model  == "knn":
                model = Pipeline([('scl', StandardScaler()),
                              ('knnr',KNeighborsRegressor(n_neighbors=num_neighbours))])
                
            elif self.model  == "svr":
                model = Pipeline([('scl', StandardScaler()),
                              ('svr',SVR(kernel=kernel))])
                
            elif self.model  == "pca":
                model = Pipeline([('pca',PCA(n_components=comp)),('lr',LinearRegression())])
                
            elif self.model  == "sgdr":
                model = make_pipeline(StandardScaler(),SGDRegressor(random_state=42))
                
            elif self.model  == "plsr1" or self.model == 'plsr2':
                model = Pipeline([('plsr',PLSRegression())])
                
        except:
            print("The model name is incorrect")
        return model
    
    def train_test_data(self,X,y,y_t,y_q,y_e):
        
        X_data = X
        if self.output  == "temperature":
            y_data = y_t
        if self.output  == "heat":
            y_data = y_q
        if self.output  == "efficiency":
            y_data = y_e
        if self.output  == "all":
            y_data = y
        return X_data,y_data
    
    def max_value(self,dict_): #max value in dictionary
        k = [key for key,value in dict_.items() if value == max(list(dict_.values()))]
        return k[0]
          
    def param_evaluate_model(self,X,y,model):
        X_train,y_train,X_val,y_val = self.evaluate_model(X,y,model)
        return X_train,y_train,X_val,y_val
    
    def plot_dictionary(self,dict_,output):
        fig = plt.figure(figsize =(10, 7)) 
        plt.plot(*zip(*sorted(dict_.items())),marker="o",color='black',linestyle='dashed',markersize=10,markerfacecolor="c")
        plt.title(f"Fig for {output} and for case {self.case}")
        plt.show()
        if not os.path.exists("Other plots"):
            os.makedirs('Other plots')
        fig.savefig(f"Other plots/{self.model}--{self.output}--{self.case}.png")
    
    def prep_data_for_PI(self,train):
        '''
        Permutation Feature Importance
        '''
        predictions,y_val,r2_score,model,error = self.train_model()
        if train:
            X,y,y_t,y_q,y_e = self.pre_processing(True)
            X,y = self.train_test_data(X,y,y_t,y_q,y_e)
            X_train,y_train,X_val,y_val = self.param_evaluate_model(X,y,model)
            return X_train,y_train,model
        else:
            X_test,y_test = self.test_data()
            return X_test,y_test,model
    
    def permutation_importance(self,train,top_limit=None):  
        
        X,y,model = self.prep_data_for_PI(train)   
        bunch = permutation_importance(model,X,y,
                                 n_repeats=50,random_state=42)
        imp_means = bunch.importances_mean

        ordered_imp_means_args = np.argsort(imp_means)[::-1]

        if top_limit is None:
            top_limit = len(ordered_imp_means_args)
        
        feature_names = ['T ambient','Intensity','Ti','Flow rate']
        imp_scores_dict = {}
        for i,_ in zip(ordered_imp_means_args,range(top_limit)):
            name = feature_names[i]
            imp_score = imp_means[i]
            imp_std = bunch.importances_std[i]
            imp_scores_dict[name] = imp_score
        return imp_scores_dict
    
    def best_param(self):
        '''
        Choosing the best k value in KNNRegressor
        The best kernel for SVM
        The best number of components for PCA
        It is brute force testing
        '''
        k_value = 0
        comp = 0
        kernel = ""
        X,y,y_t,y_q,y_e = self.pre_processing(True)
        X,y = self.train_test_data(X,y,y_t,y_q,y_e)
        if self.model  == "knn":
            scores_dict = {}
            for k in range(1,16):
                model = self.models(num_neighbours=k)
                X_train,y_train,X_val,y_val = self.param_evaluate_model(X,y,model)
                model.fit(X_train,np.ravel(y_train))
                preds = model.predict(X_val)
                r2_score = metrics.r2_score(y_val,preds)
                scores_dict[k] = r2_score
#             self.plot_dictionary(scores_dict,self.output)
            k_value = self.max_value(scores_dict)
#             print("The k value is:",k_value)
        if self.model  == 'svr':
            scores_dict = {}
            kernels = ["linear", "poly", "rbf", "sigmoid"]
            for kernel in kernels:
                model = self.models(kernel=kernel)
                X_train,y_train,X_val,y_val = self.param_evaluate_model(X,y,model) 
                model.fit(X_train,np.ravel(y_train))
                preds = model.predict(X_val)
                r2_score = metrics.r2_score(y_val,preds)
                scores_dict[kernel] = r2_score
            kernel = self.max_value(scores_dict)
#             print(f'The kernel for {self.case} and {self.output} SVM is {kernel}')
        
        if self.model  == 'pca' or self.model  == 'plsr1' or self.model == 'plsr2':
            scores_dict = {}
            for comp in range(1,5):
                model = self.models(comp=comp)
                X_train,y_train,X_val,y_val = self.param_evaluate_model(X,y,model)
                model.fit(X_train,y_train)
                preds = model.predict(X_val)
                r2_score = metrics.r2_score(y_val,preds)
                scores_dict[comp] = r2_score
#             print(scores_dict)
            comp = self.max_value(scores_dict)
#             print(f"Number of prinicipal components for {self.case} {self.output} is 4 ")
        return k_value,kernel,comp
            
        
    def evaluate_model(self,X,y,model):
        '''
        K-Fold Cross Validation
        '''
        cv_scores = {}
        
        try:
            for n in range(2,10): #self.len_data()//2+1
                cv = RepeatedKFold(n_splits = n, n_repeats=4, random_state = 42)
                r2_score = cross_val_score(model,X,y,scoring = 'r2', cv=cv, n_jobs = -1)
                cv_scores[n] = r2_score
        except Exception as e:
            print(e)
#         print(cv_scores)
        means,maxs = [],[]
        for arr in cv_scores.values():
            mean = np.mean(arr)
            max_ = np.max(arr)
            means.append(mean)
            maxs.append(max_)
        for key,values in cv_scores.items():
            for v in values:
                if v == np.max(maxs):
                    final_split = key  
#         print("The final split is",final_split)
        f_cv = RepeatedKFold(n_splits = final_split, n_repeats=4, random_state = 42)
        scores = cross_val_score(model,X,y,scoring = 'r2', cv=f_cv, n_jobs = -1)
        
        idx = list(scores).index(max(scores))
        
        train_idx,val_idx = [],[]
        for train,val in f_cv.split(X):
            train_idx.append(train)
            val_idx.append(val)
        X_train = [X.loc[i] for i in list(train_idx)[idx]]
        X_val = [X.loc[i] for i in list(val_idx)[idx]]
        if self.model  == "plsr2":
            y_train = [y.values[i] for i in list(train_idx)[idx]]
            y_val = [y.values[i] for i in list(val_idx)[idx]]    
        else :
            y_train = [y.values[i] for i in list(train_idx)[idx]]
            y_val = [y.values[i] for i in list(val_idx)[idx]]  
        
        return X_train,y_train,X_val,y_val
    
    
    def train_evaluate_model(self,X,y):
        k_value,kernel,comp = self.best_param()
        model = self.models(num_neighbours=k_value,comp=comp,kernel=kernel)
        X_train,y_train,X_val,y_val = self.evaluate_model(X,y,model)
        return X_train,y_train,X_val,y_val,model
        
    def train_model(self):
        X,y,y_t,y_q,y_e = self.pre_processing(True)
        X_data,y_data = self.train_test_data(X,y,y_t,y_q,y_e) 
        
        X_train,y_train,X_val,y_val,model = self.train_evaluate_model(X_data,y_data)
#         print(np.array(X_val).shape)
        if self.model == 'plsr2':
            model.fit(X_train,y_train)
            predictions = model.predict(X_val)
            temp_preds = []
            temp_y = []
            r2_scores = {}
            error = {}
            outputs = ["Temperature","Heat","Efficiency"]
            for col in range(3):
                for row in range(len(y_val)):
                    temp_preds.append(predictions[row][col])
                    temp_y.append(y_val[row][col])
                r2_scores[outputs[col]] = metrics.r2_score(temp_y,temp_preds)
                error[outputs[col]] = np.sqrt(metrics.mean_squared_error(temp_y,temp_preds))
                temp_preds = []
                temp_y = []
        else:
            model.fit(X_train,np.ravel(y_train))
            predictions = model.predict(X_val)
            r2_scores = metrics.r2_score(y_val,predictions)
            error = np.sqrt(metrics.mean_squared_error(y_val,predictions))
#         print(f"The predictions are {predictions} and true data are {y_val}")
#         self.plot_data(predictions,y_val,"Validation")
        return predictions,y_val,r2_scores,model,error
            
    def plot_data(self,preds,y,val_or_test):
        if val_or_test == "Test": y = y.values
        if self.model  == 'plsr2':
            for i in range(len(y)):
                plt.style.use('ggplot')
                fig,ax = plt.subplots()
                ax.plot(list(preds[i]),label="Predicted data",marker='s',linestyle='None')
                ax.plot(list(y[i]),label="Actual data",marker='o',linestyle='None')
                ax.set_title(f"Fig of {val_or_test} for {self.output} and for case {self.case}")
                ax.legend() 
#                 plt.grid(True)
                if not os.path.exists("Validation plots"):
                    os.makedirs('Validation plots')
                if val_or_test=="Validation":
                    fig.savefig(f"Validation plots/{self.output}--{self.case}--{self.model}")
        else:
            plt.style.use('ggplot')
            fig,ax = plt.subplots() 
            ax.plot(list(preds),label = "Predicted data",marker="o",markerfacecolor="red",linestyle='None')
            ax.plot(y,label="Actual data",marker="s",markerfacecolor="green",linestyle='None')
            ax.set_title(f"Fig of {val_or_test} for {self.output} and for case {self.case}")
            ax.legend()
#             plt.grid(True)
            if not os.path.exists("Validation plots"):
                os.makedirs('Validation plots')
            if val_or_test=="Validation":
                fig.savefig(f"Validation plots/{self.output}--{self.case}--{self.model}")
    
    def test_data(self):
        X,y,y_t,y_q,y_e = self.pre_processing(False)
        X_test,y_test = self.train_test_data(X,y,y_t,y_q,y_e)
        return X_test,y_test
    
    def test_model(self,model):
        X_test,y_test = self.test_data()
        test_preds = model.predict(X_test)
#         self.plot_data(test_preds,y_test,"Test")

        if self.model == 'plsr2':
            temp_preds = []
            temp_y = []
            r2_scores = {}
            outputs = ["Temperature","Heat","Efficiency"]
            for col in range(3):
                for row in range(len(y_test.values)):
                    temp_preds.append(predictions[row][col])
                    temp_y.append(y_test.values[row][col])
                r2_scores[outputs[col]] = metrics.r2_score(temp_y,temp_preds)
                temp_preds = []
                temp_y = [] 
        else:
            r2_scores = metrics.r2_score(y_test,test_preds)
        return r2_scores,y_test,test_preds
    
    def len_data(self):
        df_data,df_train,df_test = self.read_data()
        return len(df_train)
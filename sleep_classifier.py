import json, csv
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

import sklearn.model_selection
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit #for time series split of data
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

import math, statistics

import numpy as np
import pandas as pd

from datetime import datetime, timedelta
import time

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from parsing import parse, patients

from Patient_LFP import *

import os
import subprocess

PATH = "/Users/arjunbalachandar/Desktop/University of Toronto/Research/Fasano lab - next projects/Sleep DBS/AllFiles"
os.chdir(PATH)

#If doing multiple files
#with open('file_names.txt') as fn:
#    file_names = fn.readlines()

#patient_LFPs = []

#fix random seed for reproducibility of random number generator
seed = 7
np.random.seed(seed)

'''-----Functions-----'''
def average_prior_LFPs(lfps, num_prior_ave, use_bisensing,use_alpha,use_beta):
    lfp_ave = []
    if use_bisensing:
        if not use_alpha or not use_beta: #if sensing both sides but only want to use alpha or beta alone, then treat as if unisensing for averaging:
            for i in range(len(lfps)):
                row = lfps[i]
                new_row = []
                new_row.append(row[0])
                if num_prior_ave == 3.0:
                    for j in range(1,len(row)-1,1):
                        next_ave = (row[j]+row[j-1]+row[j+1])/num_prior_ave
                        new_row.append(next_ave)
                elif num_prior_ave == 5.0:
                    for j in range(2,len(row)-2,1):
                        next_ave = (row[j]+row[j-1]+row[j-2]+row[j+1]+row[j+2])/num_prior_ave
                        new_row.append(next_ave)
                lfp_ave.append(new_row)
        else:
            for i in range(len(lfps)):
                row = lfps[i]
                new_row = []
                new_row.append(row[0]) #the first 2 values of row remain same, are the value of beta/alpha at current time point
                new_row.append(row[1])
                if num_prior_ave == 3.0:
                    for j in range(2,len(row)-2,2):
                        next_ave_1 = (row[j]+row[j-2]+row[j+2])/num_prior_ave
                        next_ave_2 = (row[j+1]+row[j-1]+row[j+3])/num_prior_ave
                        new_row.append(next_ave_1)
                        new_row.append(next_ave_2)
                #0 1 2 3 4 5 6 7 8 9
                elif num_prior_ave == 5.0:
                    for j in range(4,len(row)-4,2):
                        next_ave_1 = (row[j]+row[j-2]+row[j-4]+row[j+2]+row[j+4])/num_prior_ave
                        next_ave_2 = (row[j+1]+row[j-1]+row[j-3]+row[j+3]+row[j+5])/num_prior_ave
                        new_row.append(next_ave_1)
                        new_row.append(next_ave_2)
                lfp_ave.append(new_row)
    else:
        for i in range(len(lfps)):
            row = lfps[i]
            new_row = []
            new_row.append(row[0])
            if num_prior_ave == 3.0:
                for j in range(1,len(row)-1,1):
                    next_ave = (row[j]+row[j-1]+row[j+1])/num_prior_ave
                    new_row.append(next_ave)
            elif num_prior_ave == 5.0:
                for j in range(2,len(row)-2,1):
                    next_ave = (row[j]+row[j-1]+row[j-2]+row[j+1]+row[j+2])/num_prior_ave
                    new_row.append(next_ave)
            lfp_ave.append(new_row)
    return lfp_ave

def classify_sleep(left_patient_LFP,right_patient_LFP,use_bisensing,num_train,num_split,reverse_traintest,average_priors,num_prior_ave,use_left,use_right,use_alpha,use_beta):
    #use_bisensing - if True, use both hemispheres if both sides recordings are present
    if use_bisensing:
        if left_patient_LFP.LFP_is_present and right_patient_LFP.LFP_is_present:
            #bisensing = True
            bisensing = True
    else:
        bisensing = False
    return classify_sleep_bisensing(left_patient_LFP,right_patient_LFP,bisensing,use_bisensing,num_train,num_split,reverse_traintest,average_priors,num_prior_ave,use_left,use_right,use_alpha,use_beta) #use_bisensing is True if both electrodes are sensing, bisensing is True is want to use both

def classify_sleep_bisensing(left_patient_LFP,right_patient_LFP,bisensing,use_bisensing,num_train,num_split,reverse_traintest,average_priors,num_prior_ave,use_left,use_right,use_alpha,use_beta):
    multi_class = False #if classifying wake vs ambiguous vs deepsleep/REM, as opposed to wake vs sleep
    kcrossv = False #if using k-fold cross validation
    
    #Check what specific freq band each side (or one side if unilateral sensing) is sensing
    use_alpha_r = False
    use_alpha_l = False
    use_beta_r = False
    use_beta_l = False

    if use_left:
        print(left_patient_LFP.LFP_freq_band)
        if left_patient_LFP.LFP_freq_band == 'alpha':
            use_alpha_l = True
        elif left_patient_LFP.LFP_freq_band == 'beta':
            use_beta_l = True
    if use_right:
        print(right_patient_LFP.LFP_freq_band)
        if right_patient_LFP.LFP_freq_band == 'alpha':
            use_alpha_r = True
        elif right_patient_LFP.LFP_freq_band == 'beta':
            use_beta_r = True
        
    if use_bisensing:
        time = np.array(left_patient_LFP.time)
        y_raw = left_patient_LFP.in_sleep
        y_all_labels_multi = np.array(left_patient_LFP.sleep_stage_num).astype(np.int)
    else:
        if use_alpha_r or use_beta_r:#use_right:
            time = np.array(right_patient_LFP.time)
            y_raw = right_patient_LFP.in_sleep
        else:
            time = np.array(left_patient_LFP.time)
            y_raw = left_patient_LFP.in_sleep
    y_all_labels = np.array(y_raw).astype(np.int)#convert to nparray
            
    x_alpha = []
    x_beta = []
    #print('Left Freq band: ' + left_patient_LFP.LFP_freq_band)
    if use_bisensing:
        if left_patient_LFP.LFP_freq_band == 'beta':
            x_beta = (np.array(left_patient_LFP.LFP)).astype(np.float)
            x_alpha = (np.array(right_patient_LFP.LFP)).astype(np.float)
        else:
            x_alpha = (np.array(left_patient_LFP.LFP)).astype(np.float)
            x_beta = (np.array(right_patient_LFP.LFP)).astype(np.float)
        num_values = len(x_alpha)
    else:
        if use_beta_r:
            x_beta = (np.array(right_patient_LFP.LFP)).astype(np.float)
            num_values = len(x_beta)
        elif use_alpha_r:
            x_alpha = (np.array(right_patient_LFP.LFP)).astype(np.float)
            num_values = len(x_alpha)
        elif use_beta_l:
            x_beta = (np.array(left_patient_LFP.LFP)).astype(np.float)
            num_values = len(x_beta)
        elif use_alpha_l:
            x_alpha = (np.array(left_patient_LFP.LFP)).astype(np.float)
            num_values = len(x_alpha)
        
    x_priors = []

    if bisensing and use_bisensing: #use both alpha and beta only if want to (bisensing), even if both hemispheres being sensed (use_bisensing)
        '''if use_alpha:
            print("using alpha")
        if use_beta:
            print("using beta")'''
            
        for i in range(num_train+1):
            prior_alpha = x_alpha[num_train-i:num_values-i]
            prior_beta = x_beta[num_train-i:num_values-i]
            
            if use_alpha:
                x_priors.append(prior_alpha)
            if use_beta:
                x_priors.append(prior_beta)
    else:
        for i in range(num_train+1):
            if use_beta_r or use_beta_l:
                prior_beta = x_beta[num_train-i:num_values-i]
                x_priors.append(prior_beta)    
            elif use_alpha_r or use_alpha_l:
                prior_alpha = x_alpha[num_train-i:num_values-i] #IF USING ONLY ALPHA
                x_priors.append(prior_alpha)

    x_all = np.transpose(x_priors)
    
    #if using averaging method of priors and not raw priors themselves
    if average_priors:
        x_all = average_prior_LFPs(x_all, num_prior_ave, bisensing,use_alpha,use_beta)

        
    y_all_labels = y_all_labels[num_train:num_values]

    
    #for multi-class classification problem
    '''
    if bisensing:
        for i in range(len(y_all_labels_multi)):
            if (y_all_labels_multi[i] == 3 or y_all_labels_multi[i] == 4 or y_all_labels_multi[i] == 5):
            #if (y_all_labels_multi[i] == 3 or y_all_labels_multi[i] == 4):
                #y_all_labels_multi[i] = 2
                y_all_labels_multi[i] = 2
            elif (y_all_labels_multi[i] == 2):
                y_all_labels_multi[i] = 1
            else:
                y_all_labels_multi[i] = 0
        y_all_labels_multi = y_all_labels_multi[num_train:num_values]
    '''
    
    #x_all_modified = np.stack((x_alpha, x_beta, y_all_labels), axis=1) #includes y_label in training features, since we'll use the preceding num_train
    
    kfold = KFold(n_splits=num_split, shuffle=False)# random_state=seed)
    cvscores = [] #keeps track of each fold's accuracy
    
    y_pred = []
    y_tot_test = [] #list of all labels actually used for training/testing, since some are ignored because of preventing overlap in training and testing

    #model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0)
    #model = LogisticRegression(random_state=0, solver='lbfgs',multi_class='ovr')
    #model = LogisticRegression(random_state=0)
    #model = GaussianProcessClassifier(1.0 * RBF(1.0))
    #model = GaussianNB()
    #model = SVC(kernel='rbf',gamma=2, C=1)
    #model = DecisionTreeClassifier(random_state=0)
    #model = KNeighborsClassifier()
    #model = MLPClassifier(hidden_layer_sizes=(100,100,100,100,100,100,100),random_state=1, max_iter=5000)
    #model = MLPClassifier(hidden_layer_sizes=(500,500,500,500),random_state=1)
    
    ''''
    if multi_class:
        y_all_labels = y_all_labels_multi
    '''
    
    if kcrossv:
        num_test = int(len(y_all_labels)/num_split) #length of training set
        num_tot = len(y_all_labels)
        for i in range(num_split):
            if i==0: #in this case, already removed firsy num_train values from list at onset earlier
                x_test = x_all[0:num_test]
                y_test = y_all_labels[0:num_test]
                x_train = x_all[num_test+num_train:len(y_all_labels)]
                y_train = y_all_labels[num_test+num_train:len(y_all_labels)]
            elif i==(num_split-1):
                x_train = x_all[0:(len(y_all_labels) - num_test)]
                y_train = y_all_labels[0:(len(y_all_labels) - num_test)]
                x_test = x_all[(len(y_all_labels) - num_test + num_train):len(y_all_labels)]
                y_test = y_all_labels[(len(y_all_labels) - num_test + num_train):len(y_all_labels)]
            else:
                x_test = x_all[(num_test*i+num_train):(num_test*(i+1))]
                y_test = y_all_labels[(num_test*i+num_train):(num_test*(i+1))]
                x_train = [*x_all[0:(num_test*i)],*x_all[(num_test*(i+1)+num_train):len(y_all_labels)]]
                y_train = [*y_all_labels[0:(num_test*i)],*y_all_labels[(num_test*(i+1)+num_train):len(y_all_labels)]]
            #train model
            clf = model.fit(x_train,y_train)
            pred = clf.predict(x_test)
            #overall accuracy
            scores = clf.score(x_test,y_test)
            cvscores.append(scores * 100)
            #concatonate to total list of predictions/correct
            y_pred = [*y_pred,*pred]
            y_tot_test = [*y_tot_test,*y_test]
            
        print(classification_report(y_tot_test, y_pred))
        print('Confusion Matrix')
        con_matrix = confusion_matrix(y_tot_test, y_pred)
        print(con_matrix)
        print('Total Accuracy: '+str(np.mean(cvscores)))
        AUC = roc_auc_score(y_tot_test, y_pred)
        print('AUC: '+str(AUC))
    else:
        num_test = int(len(y_all_labels)/num_split)
        if reverse_traintest:
            x_test = x_all[0:num_test] #x_all[0:(len(y_all_labels) - num_test)]
            y_test = y_all_labels[0:num_test]#y_all_labels[0:(len(y_all_labels) - num_test)]
            x_train = x_all[(num_test + num_train):len(y_all_labels)]
            y_train = y_all_labels[(num_test + num_train):len(y_all_labels)]
        else:
            x_train = x_all[0:(len(y_all_labels) - num_test)]
            y_train = y_all_labels[0:(len(y_all_labels) - num_test)]
            x_test = x_all[(len(y_all_labels) - num_test + num_train):len(y_all_labels)]
            y_test = y_all_labels[(len(y_all_labels) - num_test + num_train):len(y_all_labels)]

        print(y_test)
        '''
        print(num_test)
        print(len(x_all))
        print(len(x_train))
        print(len(x_test))
        print(len(y_train))
        print(len(y_test))
        '''

        clf = model.fit(x_train,y_train)

        y_pred = clf.predict(x_test)
        print(y_pred)
        scores = clf.score(x_test,y_test)
        print('Scores:')
        print(scores)

        print(classification_report(y_test, y_pred))
        print('Confusion Matrix')
        con_matrix = confusion_matrix(y_test, y_pred)
        print(con_matrix)
        #print('Accuracy: '+str(np.mean(cvscores)))

        #ROC
        AUC = roc_auc_score(y_test, y_pred)
        print('AUC: '+str(AUC))
        
        return [y_test,y_pred]
    
    '''
    #for train, test in kfold.split(np.zeros(y_all_labels.size), y_all_labels):
        #model.fit(x_all[train],y_all_labels[train])
    #for train, test in cv.split(np.zeros(y_all_labels.size), y_all_labels):
    clf = model.fit(x_all[train],y_all_labels[train])
        
        pred = clf.predict(x_all[test])
        scores = clf.score(x_all[test], y_all_labels[test])
        #print(scores)
        cvscores.append(scores * 100)
        tot_cvscores.append(scores * 100)
        y_pred.append(pred[0])
        print(time[test][0]+" - Actual: "+str(y_all_labels[test][0])+" - Pred: "+str(pred[0]))
    
    #Y_test = Y_all_labels #np.argmax(y_all, axis=1)
    print(classification_report(y_all_labels, y_pred))
    print('Confusion Matrix')
    con_matrix = confusion_matrix(y_all_labels, y_pred)
    print(con_matrix)
    print('Accuracy: '+str(np.mean(cvscores)))

    #ROC
    AUC = roc_auc_score(y_all_labels, y_pred)
    print('AUC: '+str(AUC))
    '''

#ROC AUC - function used to generate area under the curve of ROC, using inputs of predicted vs. correct category
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

#one-hot encoding: convert data representation to one-hot encoding, which is required for inputs to train machine learning classifiers in Scikit-learn
def to_one_hot(labels, dimension):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, int(label)-1] = 1.
    return results

def openfile_and_classify(fn,num_train,num_split,reverse_traintest,average_priors,num_prior_ave,use_alpha,use_beta):
    new_path = PATH + "/" + fn
    os.chdir(new_path)
    left_LFP_file = fn + 'left_LFP.csv'
    right_LFP_file = fn + 'right_LFP.csv'
    #Left-hemisphere LFP file
    
    use_left = True
    use_right = True

    patient_left_LFP = []
    patient_right_LFP = []
    use_bisensing = True #use both hemisphere by default, becomes false if one side doesnt exist
    try:
        with open(left_LFP_file, mode ='r') as f:
            reader = csv.reader(f)
            LFP_is_present = True
            row_num = 0 #counter of which row in file currently in, first row is labels
            LFP = []
            time = []
            in_sleep = []
            sleep_stage_num = []
            LFP_freq = 0
            LFP_freq_band = 'beta'
            for i in reader:
                if row_num > 0:
                    time.append(i[0])
                    LFP.append(i[1])
                    sleep_stage_num.append(i[2])
                    in_sleep.append(i[3])
                    LFP_freq_band = i[5]
                    LFP_freq = i[6]
                row_num += 1
            patient_left_LFP = Patient_LFP(int(fn), time, LFP, sleep_stage_num, in_sleep, LFP_freq_band, LFP_freq,LFP_is_present)
    except FileNotFoundError:
        use_left = False
        use_bisensing = False
        print('File: '+ left_LFP_file +' does not exist')

    #Right-hemisphere LFP file
    try:
        with open(right_LFP_file, mode ='r') as f:
            reader = csv.reader(f)
            LFP_is_present = True
            row_num = 0 #counter of which row in file currently in, first row is labels
            LFP = []
            time = []
            in_sleep = []
            sleep_stage_num = []
            LFP_freq = 0
            LFP_freq_band = 'beta'
            for i in reader:
                if row_num > 0:
                    time.append(i[0])
                    LFP.append(i[1])
                    sleep_stage_num.append(i[2])
                    in_sleep.append(i[3])
                    LFP_freq_band = i[5]
                    LFP_freq = i[6]
                row_num += 1
            patient_right_LFP = Patient_LFP(int(fn), time, LFP, sleep_stage_num, in_sleep, LFP_freq_band, LFP_freq,LFP_is_present)
    except FileNotFoundError:
        use_right = False
        use_bisensing = False
        print('File: '+ right_LFP_file +'does not exist')
    return classify_sleep(patient_left_LFP,patient_right_LFP,use_bisensing,num_train,num_split,reverse_traintest,average_priors,num_prior_ave,use_left,use_right,use_alpha,use_beta)


'----Main Code-----'
fn = '008'
combineResults = False #get total confusion matrix across multiple patients if True
reverse_traintest = False
average_priors = False #average n-prior LFP power data points into one value, for minimizing number of training features and reduce noise
num_prior_ave = 5.0 #number of LFP power data points to average into one value, for minimizing number of training features and reduce noise
num_train = 80 #number of data points preceding current test point to use for training, including their labels
num_split = 3 #for train/test split

use_alpha = False #if bilateral sensing and want to just use alpha or beta, can make either False
use_beta = True

openfile_and_classify(fn,num_train,num_split,reverse_traintest,average_priors,num_prior_ave,use_alpha,use_beta)
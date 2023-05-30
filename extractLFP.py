#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 13:49:17 2022

@author: arjunbalachandar
"""
import json, csv

import os
import subprocess

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter

import math, statistics
from statistics import mean, stdev

import numpy as np
import pandas as pd

from datetime import datetime, timedelta
import time

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from parsing import parse, patients

PATH = "/Users/arjunbalachandar/Desktop/University of Toronto/Research/Fasano lab - next projects/Sleep DBS/AllFiles"

def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

def extractData(p):
    fn = p.patient_num
    new_path = PATH + "/" + fn
    os.chdir(new_path)
    
    #if sensing right and left hemisphere, these become True respectively
    right_present = False
    left_present = False
    
    outlier_inds_left = []
    outlier_inds_right = []
    data_left = []
    data_right = []
    if (len(p.left_diagnostic_data) != 0):
        left_present = True
        data_left, outlier_inds_left = extractLFP(p, p.left_diagnostic_data, 'left')
        #print(str(fn)+' - Outlier Left')
    
    if (len(p.right_diagnostic_data) != 0):
        right_present = True
        data_right, outlier_inds_right = extractLFP(p, p.right_diagnostic_data, 'right')
        #print(str(fn)+' - Outlier Right')
    
    #combined list of outliers from left and right hemispheres, removing overlap indices
    #print(outlier_inds_left)
    if (left_present and right_present):
        outlier_inds_net = sorted(np.unique(outlier_inds_left + outlier_inds_right))
        
        '''
        if (fn=='002'):
            print('Pre-outlier')
            print(data_left[3])
            print(len(data_left[3]))
        '''
        
        for i in range(len(data_left)-2):
            delete_multiple_element(data_left[i],outlier_inds_net)
            delete_multiple_element(data_right[i],outlier_inds_net)
        max_lfp_left = max(data_left[1])
        max_lfp_right = max(data_right[1]) #for normalizing
        
        ''' for checking correctness, to delete
        print(str(fn)+' - Outlier Net Length')
        #print(outlier_inds_net)
        print(len(outlier_inds_net))
        '''
        
        '''
        if (fn=='002'):
            print('Post-outlier')
            print(data_left[3])
            print(len(data_left[3]))
        '''
        
        #Write rows to corresponding files
        fields = ['time','LFP power','stage (number)','wake (0) vs sleep (1)','stage (full)','LFP freq band','LFP freq (Hz)']
        
        left_file = fn + '' + 'left_LFP.csv'
        right_file = fn + '' + 'right_LFP.csv'
        f_left = open(left_file,'w')
        f_right = open(right_file,'w')
        
        writer_left = csv.writer(f_left)
        writer_right = csv.writer(f_right)
        writer_left.writerow(fields)
        writer_right.writerow(fields)
        
        for i in range(len(data_left[1])):
            #NORMALIZED DATA IS WRITTEN TO FILE
            row_left = [data_left[0][i],(1.0*data_left[1][i])/(1.0*max_lfp_left),data_left[2][i],data_left[3][i],data_left[4][i],data_left[5],data_left[6]]
            #print(row_left)
            row_right = [data_right[0][i],(1.0*data_right[1][i])/(1.0*max_lfp_right),data_right[2][i],data_right[3][i],data_right[4][i],data_right[5],data_right[6]]
            writer_left.writerow(row_left)
            writer_right.writerow(row_right)
        f_left.close()
        f_right.close()
    else:
        if (left_present and not right_present): #If only left hemisphere recording
            for i in range(len(data_left)-2):
                delete_multiple_element(data_left[i],outlier_inds_left)
            max_lfp_left = max(data_left[1]) #for normalizing
            #Write rows to corresponding files
            fields = ['time','LFP power','stage (number)','wake (0) vs sleep (1)','stage (full)','LFP freq band','LFP freq (Hz)']
            
            left_file = fn + '' + 'left_LFP.csv'
            f_left = open(left_file,'w')
            
            writer_left = csv.writer(f_left)
            writer_left.writerow(fields)
            
            for i in range(len(data_left[0])): #If only right hemisphere recording
                row_left = [data_left[0][i],(1.0*data_left[1][i])/(1.0*max_lfp_left),data_left[2][i],data_left[3][i],data_left[4][i],data_left[5],data_left[6]]
                writer_left.writerow(row_left)
            f_left.close()
        elif (right_present and not left_present):
            for i in range(len(data_right)-2):
                delete_multiple_element(data_right[i],outlier_inds_right)
            max_lfp_right = max(data_right[1]) #for normalizing
            #Write rows to corresponding files
            fields = ['time','LFP power','stage (number)','wake (0) vs sleep (1)','stage (full)','LFP freq band','LFP freq (Hz)']
            
            right_file = fn + '' + 'right_LFP.csv'
            f_right = open(right_file,'w')
            
            writer_right = csv.writer(f_right)
            writer_right.writerow(fields)
            
            for i in range(len(data_right[0])):
                row_right = [data_right[0][i],(1.0*data_right[1][i])/(1.0*max_lfp_right),data_right[2][i],data_right[3][i],data_right[4][i],data_right[5],data_right[6]]
                writer_right.writerow(row_right)
            f_right.close()

    '''
    with open(new_file, 'w') as csvfile2:
        csvwriter2 = csv.writer(csvfile2)
        csvwriter2.writerow(fields)
        max_lfp_left = max(data_left[1])
        max_lfp_right = max(data_right[1])
        for i in range(len(data_left[1])):
            data_left[1][i] = (1.0*data_left[1][i])/max_lfp_left
            data_right[1][i] = (1.0*data_right[1][i])/max_lfp_right
            row = [time[i],norm_lfp,all_lfp_label[i],all_lfp_insleep_label[i],all_lfp_label_full[i],LFP_freq_band,LFP_freq]
    '''

    
def extractLFP(p, diagnostic_data, hemisphere): #returns extracted data, and list of indexes of outliers to be removed later
    wake_lfp = []
    restless_lfp = []
    light_lfp = []
    deep_lfp = []
    rem_lfp = []
    all_lfp = [] #all data points
    all_lfp_label_full = [] #the label for each LFP point by sleep-stage
    all_lfp_label = []
    all_lfp_insleep_label = []
    time = []
    outlier_inds = [] #list of indexes of outliers, so that later can remove elements at this index
    
    rows = []
    all_data = []
    
    label = 'wake'
    label_num = 0 #LFP Labels: 0=wake, 1=restless, 2=light, 3=deep, 4=rem
    in_sleep = 0 #for binary wake vs sleep , wake = 0, sleep = 1
    freq = 0
    
    wake_lfp_sum, restless_lfp_sum, light_lfp_sum, deep_lfp_sum, rem_lfp_sum = 0, 0, 0, 0, 0
    wake_n, restless_n, light_n, deep_n, rem_n = 0, 0, 0, 0, 0
    all_n = 0
    all_sum = 0
    
    if hemisphere == 'left':
        LFP_freq_band = p.left_freqBand
        LFP_freq = p.left_freq
    else:
        LFP_freq_band = p.right_freqBand
        LFP_freq = p.right_freq
    
    fn = p.patient_num

    
    for d in diagnostic_data:
        in_stage = False
        in_sleep = 0
        label = 'wake'
        label_num = 0
        in_sleep = 0 #for binary wake vs sleep , wake = 0, sleep = 1
        for stage in p.stage_starttimes_endtimes:
            '''if fn=='002' and d[0] >= 1656671734.0 and d[0] <= 1656725137.0:
                    print(stage.data)'''
            for t in stage.data:
                start = datetime.strptime(t['dateTime'].replace('T', ' ').replace('Z', '').split('.')[0], '%Y-%m-%d %H:%M:%S').timestamp()
                end = start + t['seconds']

                try:
                    last_stage = current_stage
                except:
                    print('first stage')
                current_stage = t['level']
                
                #LFP Labels: 0=wake, 1=restless, 2=light, 3=deep, 4=rem
                if d[0]>=start and d[0]<=end:
                    in_stage = True
                    #if fn=='002':
                    #    print('CURRENT STAGE: '+current_stage)
                    if current_stage == 'wake' or current_stage == 'awake':
                        wake_lfp_sum += d[1]
                        wake_n += 1
                        wake_lfp.append(d[1])
                        
                        label = 'wake'
                        label_num = 0
                        in_sleep = 0 #for binary wake vs sleep , wake = 0, sleep = 1
                    elif current_stage != 'wake' and current_stage != 'awake':
                        in_sleep = 1 #for binary wake vs sleep , wake = 0, sleep = 1
                        if current_stage == 'restless':
                            label = 'restless'
                            label_num = 1
                        elif current_stage == 'light':
                            label = 'light'
                            label_num = 2
                        elif current_stage == 'deep':
                            label = 'deep'
                            label_num = 3
                        elif current_stage == 'rem':
                            label = 'rem'
                            label_num = 4
                        elif current_stage == 'asleep':
                            label = 'asleep'
                            label_num = 5
        all_lfp.append(d[1])
        all_lfp_label_full.append(label)
        all_lfp_label.append(label_num)
        all_lfp_insleep_label.append(in_sleep)
        time.append(d[0])
        #add to csv file
        #if in_stage:
        if fn=='002' and label=='rem':
            row = [d[0],d[1],label_num,in_sleep,label]
            print(row)
        #csvwriter2.writerow(row)
    
    #max_all_lfp = max(all_lfp)
    #norm_lfp = 0
    #Export data to individual files, and normalize using max value as above
    
    '''For outlier detection'''
    all_data = [time,all_lfp,all_lfp_label,all_lfp_insleep_label,all_lfp_label_full,LFP_freq_band,LFP_freq]
    '''
    if fn=='002':
        print(len(time))
    '''
    '''
    mean_all_lfp = mean(all_lfp)
    stdev_all_lfp = stdev(all_lfp,mean_all_lfp)
    upper_lim = mean_all_lfp + 3.0*stdev_all_lfp
    lower_lim = mean_all_lfp - 3.0*stdev_all_lfp
    
    ypbot = np.percentile(all_lfp, 1)
    yptop = np.percentile(all_lfp, 99)
    ypad = 0.05*(yptop - ypbot)
    lower_lim = ypbot - ypad
    upper_lim = yptop + ypad
    '''
    q25 = np.quantile(all_lfp,0.25)
    q75 = np.quantile(all_lfp,0.75)
    iqr = q75 - q25
    upper_lim = q75 + 2.0*iqr
    lower_lim = q25 - 1.5*iqr
    
    for i in range(len(all_lfp)):
        #norm_lfp = (100.0*all_lfp[i])/max_all_lfp
        #row = [time[i],norm_lfp,all_lfp_label[i],all_lfp_insleep_label[i],all_lfp_label_full[i],LFP_freq_band,LFP_freq]
        #rows.append(row)
        if (all_lfp[i] > upper_lim) or (all_lfp[i] < lower_lim):
            outlier_inds.append(i)
        #csvwriter2.writerow(row)

    return all_data, outlier_inds

#Run Main Program
with open('file_names.txt') as fn:
    file_names = fn.readlines()
    
for fn in file_names:
    parse(fn.split('\n')[0])
    
for p in patients:
    extractData(p)
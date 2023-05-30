#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 10:24:00 2022
patient.py
This code defines a class Patient that stores all information for a given patient, including intracrnial LFP and FitBit actigraph sleep recording data, and patient demographic information
@author: arjunbalachandar
"""

#from asyncio.windows_events import NULL
from contextlib import nullcontext
import imp

patients = []

class Patient:
    def __init__(self, first_name, last_name, gender, diagnosis, implant_date):
        self.first_name = first_name
        self.last_name = last_name
        self.gender = gender
        self.diagnosis = diagnosis
        self.implant_date = implant_date
        
        self.left_diagnostic_data = []
        self.right_diagnostic_data = []
        self.graphs = []
        
        self.patient_num = '' #The number based on file name

        self.starttimes_endtimes = []
        self.stage_starttimes_endtimes = []

        self.analysis_rows = []
        
        #Sensing frequency information for each side
        self.left_freq = 0.0
        self.right_freq = 0.0
        self.left_freqBand = ''
        self.right_freqBand = ''

        patients.append(self)

    def __repr__(self):
        return 'first name: %s, last name: %s, gender: %s, diagnosis: %s, implant date: %s' % (self.first_name, self.last_name, self.gender, self.diagnosis, self.implant_date)

class Stages:
    def __init__(self, start_time, end_time, data, main_sleep):
        self.start_time = start_time
        self.end_time = end_time

        self.data = data

        self.main_sleep = main_sleep
    
    def __repr__(self):
        return 'start time: %s, end time: %s, main sleep: %s' % (self.start_time, self.end_time, self.main_sleep)
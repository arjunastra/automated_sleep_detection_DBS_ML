'''
Created July 2023
patient.py
This code defines a class Patient that stores all information for a given patient, including intracranial LFP and FitBit actigraph sleep recording data, and patient demographic information.
'''

from asyncio.windows_events import NULL
from contextlib import nullcontext
import imp

patients = [] # Global list to store all patient objects created

class Patient:
    '''Class to create patient objects which contain patient information, demographic details, diagnostic data, sleep recording data, and sensing frequency information.'''
    def __init__(self, first_name, last_name, gender, diagnosis, implant_date):
        self.first_name = first_name
        self.last_name = last_name
        self.gender = gender
        self.diagnosis = diagnosis
        self.implant_date = implant_date
        
        self.left_diagnostic_data = []
        self.right_diagnostic_data = []
        self.graphs = []
        
        self.patient_num = ''

        self.starttimes_endtimes = []
        self.stage_starttimes_endtimes = []

        self.analysis_rows = []
        
        self.left_freq = 0.0
        self.right_freq = 0.0
        self.left_freqBand = ''
        self.right_freqBand = ''

        patients.append(self)

    def __repr__(self):
        '''Returns a string representation of the patient object, used for debugging.'''
        return 'first name: %s, last name: %s, gender: %s, diagnosis: %s, implant date: %s' % (self.first_name, self.last_name, self.gender, self.diagnosis, self.implant_date)

class Stages:
    '''Class to store and manage sleep stage information, including start and end times, the associated data, and whether the stage is considered the main sleep period.'''
    def __init__(self, start_time, end_time, data, main_sleep):
        self.start_time = start_time
        self.end_time = end_time

        self.data = data

        self.main_sleep = main_sleep
    
    def __repr__(self):
        '''Returns a string representation of the stage object, used for debugging.'''
        return 'start time: %s, end time: %s, main sleep: %s' % (self.start_time, self.end_time, self.main_sleep)
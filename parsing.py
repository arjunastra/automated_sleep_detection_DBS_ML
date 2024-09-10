'''
Created July 2023
parsing.py
This code defines functions that extract LFP and FitBit recording data and store them using patient class.
'''

# from asyncio.windows_events import NULL
import json, csv
from datetime import datetime, timedelta
from logging import NullHandler
from patient import *
import os
import subprocess

# Set the working directory to the path containing the data
PATH = ""
os.chdir(PATH)

patients = []

# Assigning the threshold frequency (Hz)
# Below freq_threshold is alpha, above freq_threshold is beta
freq_threshold = 12.5 

def convert(t):
    '''Converts time from deep brain stimulation recordings from UTC to ET'''
    if t[-2:] == 'AM' and t[:2] == '12':
        return '00' + t[2:-2] 
    elif t[-2:] == 'AM':
        if len(t) == 7:
            return t[:-2]
        else: 
            return '0' + t[:-2]
    elif t[-2:] == 'PM' and t[:2] == '12':
        return t[:-2]   
    else:
        if len(t) == 7:
            return str(int(t[:2]) + 12) + t[2:5]
        else:
            return str(int(t[:1]) + 12) + ':' + t[2:4]

def parse(fn):
    '''Loads and parses files for a patient'''
    new_path = PATH + "/" + fn
    os.chdir(new_path)
    
    subjectFile = fn + 'subject.json' # DBS recording
    sleepFile = fn + 'sleep.csv'      # FitBit recording data with sleep start and end times only
    stageFile = fn + 'stages.json'    # FitBit recording data with sleep stages
    
    starttimes_endtimes = []
    stage_starttimes_endtimes = []

    # Read subjectFile
    with open(subjectFile) as suf:
        subjectFile = json.load(suf)

    # Read and parse sleepFile for main sleep
    with open(sleepFile, mode ='r') as slf:
        sleepFile = csv.reader(slf)
        for i in sleepFile:
            if not i:
                pass
            elif i[0] == 'Sleep' or i[0] == 'Start Time':
                pass
            else:
                start_time = (datetime.strptime(i[0].split(' ')[0] + ' ' + convert(i[0].split(' ')[1]), '%Y-%m-%d %H:%M') + timedelta(hours=0)).timestamp()
                end_time = (datetime.strptime(i[1].split(' ')[0] + ' ' + convert(i[1].split(' ')[1]), '%Y-%m-%d %H:%M') + timedelta(hours=0)).timestamp()
                starttimes_endtimes.append([start_time, end_time])

    # Read stageFile for sleep stages
    with open(stageFile) as stf:
        stageFile = json.load(stf) 
    # Parse sleep stages    
    for i in stageFile:
        # If you want to exclude anything labelled as not main sleep uncomment the following line
        # if (i['mainSleep'] == True):
            start_time = datetime.strptime(i['startTime'].replace('T', ' ').replace('Z', '').split('.')[0], '%Y-%m-%d %H:%M:%S').timestamp()
            end_time = datetime.strptime(i['endTime'].replace('T', ' ').replace('Z', '').split('.')[0], '%Y-%m-%d %H:%M:%S').timestamp()
            stage = Stages(start_time, end_time, i['levels']['data'], i['mainSleep'])
            stage_starttimes_endtimes.append(stage)

    # Parse subjectFile for patient demographic information, DBS settings, and DBS LFP recordings 
    for i in subjectFile:
        if i == 'PatientInformation':
            v = subjectFile[i]['Initial']
            patient = Patient(v['PatientFirstName'], v['PatientLastName'], v['PatientGender'], v['Diagnosis'], 0)
            patient.patient_num = fn
        elif i == 'DeviceInformation':   
            v = subjectFile[i]['Initial']
            patient.implant_date = v['ImplantDate']
        elif i == 'EventSummary':
            hemisphere_1, hemisphere_2 = None, None
            left, right = None, None
            try: 
                hemisphere_1 =  subjectFile[i]['LfpAndAmplitudeSummary'][0]['Hemisphere'] 
                if ('Left' in hemisphere_1):
                    left = hemisphere_1
                elif ('Right' in hemisphere_1):
                    right = hemisphere_1
            except IndexError:
                print('No hemisphere sensing')
            try:
                hemisphere_2 =  subjectFile[i]['LfpAndAmplitudeSummary'][1]['Hemisphere']
                if ('Left' in hemisphere_2):
                    left = hemisphere_2
                elif ('Right' in hemisphere_2):
                    right = hemisphere_2
            except IndexError:
                print('No hemisphere sensing')
        elif i == 'DiagnosticData':
            if (left != None):
                v = subjectFile[i]['LFPTrendLogs'][left] 
                for v1 in v:
                    for v2 in range(0,len(v[v1])): 
                        date_time = datetime.strptime(v[v1][v2]['DateTime'].replace('T', ' ').replace('Z', ''), '%Y-%m-%d %H:%M:%S')
                        left_diagnostic_data = [date_time.timestamp(), v[v1][v2]['LFP'], False, date_time]
                        patient.left_diagnostic_data.append(left_diagnostic_data)
                patient.left_diagnostic_data = sorted(patient.left_diagnostic_data, key=lambda diagnostic_data: diagnostic_data[0])
            if (right != None):
                v = subjectFile[i]['LFPTrendLogs'][right] 
                for v1 in v:
                    for v2 in range(0,len(v[v1])):
                        date_time = datetime.strptime(v[v1][v2]['DateTime'].replace('T', ' ').replace('Z', ''), '%Y-%m-%d %H:%M:%S')
                        right_diagnostic_data = [date_time.timestamp(), v[v1][v2]['LFP'], False, date_time]
                        patient.right_diagnostic_data.append(right_diagnostic_data)
                patient.right_diagnostic_data = sorted(patient.right_diagnostic_data, key=lambda diagnostic_data: diagnostic_data[0])
            # Update patient object variables
            patient.starttimes_endtimes = starttimes_endtimes
            patient.stage_starttimes_endtimes = stage_starttimes_endtimes
            patient.first_name = str(int(fn))
            patient.last_name = ''
            patients.append(patient)
            
            
        # Extract frequency band sensing data
        elif i == 'Groups':
            print(fn + "\n")
            a = subjectFile[i]['Initial']

            if (len(a) == 3):
                v = a[2]['ProgramSettings']['SensingChannel']
            elif (len(a) == 2):
                v = a[1]['ProgramSettings']['SensingChannel']
            else:
                v = a[0]['ProgramSettings']['SensingChannel']

            num_sensing = len(v)
            if (num_sensing >= 2):
                # Store channel 0 frequency
                if (v[0]['HemisphereLocation'] == 'HemisphereLocationDef.Left'):
                    patient.left_freq = v[0]['SensingSetup']['FrequencyInHertz']
                elif (v[0]['HemisphereLocation'] == 'HemisphereLocationDef.Right'):
                    patient.right_freq = v[0]['SensingSetup']['FrequencyInHertz']
                      
                # Store channel 1 frequency
                if (v[1]['HemisphereLocation'] == 'HemisphereLocationDef.Left'):
                    patient.left_freq = v[1]['SensingSetup']['FrequencyInHertz']
                elif (v[1]['HemisphereLocation'] == 'HemisphereLocationDef.Right'):
                    patient.right_freq = v[1]['SensingSetup']['FrequencyInHertz']
            elif (num_sensing == 1):
                if (v[0]['HemisphereLocation'] == 'HemisphereLocationDef.Left'):
                    patient.left_freq = v[0]['SensingSetup']['FrequencyInHertz']
                elif (v[0]['HemisphereLocation'] == 'HemisphereLocationDef.Right'):
                    patient.right_freq = v[0]['SensingSetup']['FrequencyInHertz']

            # Define if sensing alpha or beta
            if (patient.left_freq >= freq_threshold):
                patient.left_freqBand = 'beta'
            else:
                patient.left_freqBand = 'alpha'
            
            if (patient.right_freq > freq_threshold):
                patient.right_freqBand = 'beta'
            else:
                patient.right_freqBand = 'alpha'
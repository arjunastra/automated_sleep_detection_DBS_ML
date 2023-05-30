#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 00:39:00 2022

@author: arjunbalachandar
"""

class Patient_LFP:
    def __init__(self, patient_num, time, LFP, sleep_stage_num, in_sleep, LFP_freq_band, LFP_freq, LFP_is_present):
        self.patient_num = patient_num
        self.time = time
        self.LFP = LFP
        self.sleep_stage_num = sleep_stage_num
        self.in_sleep = in_sleep
        self.LFP_freq_band = LFP_freq_band
        self.LFP_freq = LFP_freq
        self.LFP_is_present = LFP_is_present
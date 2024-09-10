'''
Created July 2023
statistics.py
This code calculates statistics for groups based on diagnosis and target, which are written to csv files for analysis.
'''

import csv
import math
from scipy.stats import differential_entropy

# File names are based on subject number and recording hemisphere
file_names = ['002left_LFP', '002right_LFP', '003right_LFP', '004left_LFP', '004right_LFP', '005right_LFP', '006left_LFP', '006right_LFP', '007left_LFP', '007right_LFP', '008left_LFP', '009left_LFP', '010left_LFP', '010right_LFP']

# Lists for groups based on disease and sensing frequency
complete_list = []
allPD_alpha = []
allPD_beta = []
allPD_STN_beta = []
allPD_STN_alpha = []
allPD_GPI_beta = []
allPD_GPI_alpha = []
allNonPD_alpha = []
allNonPD_STN_alpha = []
allNonPD_GPI_alpha = []
allET_alpha = []

# Functions for calculating statistical measures (mean, standard deviation or variance, and differential entropy)
def calculate_mean(stage, stage_name):
    try:
        return sum(float(row[1]) for row in stage) / len(stage)
    except Exception as e:
        print(f"Error calculating mean for {stage_name}: {e}")
        return None

def calculate_std(stage, mean, stage_name):
    try:
        return (sum((float(row[1]) - mean)**2 for row in stage) / len(stage))**0.5
    except Exception as e:
        print(f"Error calculating standard deviation for {stage_name}: {e}")
        return None

def calculate_variance(std, stage_name):
    try:
        return std**2
    except Exception as e:
        print(f"Error calculating variance for {stage_name}: {e}")
        return None

def calculate_differential_entropy(lfps, stage_name):
    try:
        return differential_entropy(lfps)
    except Exception as e:
        print(f"Error calculating differential entropy for {stage_name}: {e}")
        return None

# Returns the frequency of the LFP of the stage
def extract_lfps(stage):
    return [float(row[1]) for row in stage]

# Compare stage mean with the wake mean
def compare_means(stage_mean, wake_mean, stage_name):
    try:
        if stage_mean > wake_mean:
            return 'greater'
        elif stage_mean < wake_mean:
            return 'less'
        else:
            return 'equal'
    except Exception as e:
        print(f"Error comparing {stage_name} mean with wake mean: {e}")
        return None

# Returns the difference between a stage mean and the wake mean
def calculate_difference(stage_mean, wake_mean, stage_name):
    try:
        return stage_mean - wake_mean
    except Exception as e:
        print(f"Error calculating difference for {stage_name}: {e}")
        return None

# Iterates through each file from the file_names list and calculates statistics for sleep stages
for file_name in file_names:
    current_csv = file_name + '.csv'
    subject_n = int(current_csv[0:3])

    with open(current_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        data = list(reader)

    frequency_band = data[0][5]

    wake = restless = light = deep = rem = sleepAdd = []

    for row in data:
        if row[2] == '0':
            wake.append(row)
        elif row[2] == '1': 
            restless.append(row)
        elif row[2] == '2':
            light.append(row)
        elif row[2] == '3':
            deep.append(row)
        elif row[2] == '4':
            rem.append(row)
        elif row[2] == '5':
            sleepAdd.append(row)

    sleep = restless + light + deep + rem + sleepAdd

    # Initialize variables with 'NULL'
    wake_mean = wake_std = wake_var = wake_entropy = 'NULL'
    restless_mean = restless_comparison = restless_difference = restless_std = restless_var = restless_entropy = 'NULL'
    light_mean = light_comparison = light_difference = light_std = light_var = light_entropy = 'NULL'
    deep_mean = deep_comparison = deep_difference = deep_std = deep_var = deep_entropy = 'NULL'
    rem_mean = rem_comparison = rem_difference = rem_std = rem_var = rem_entropy = 'NULL'
    sleep_mean = sleep_comparison = sleep_difference = sleep_std = sleep_var = sleep_entropy = 'NULL'

    # Calculate statistics for each stage
    # Mean
    wake_mean = calculate_mean(wake, "wake")
    restless_mean = calculate_mean(restless, "restless")
    light_mean = calculate_mean(light, "light")
    deep_mean = calculate_mean(deep, "deep")
    rem_mean = calculate_mean(rem, "REM")
    sleep_mean = calculate_mean(sleep, "sleep")

    # Standard deviation
    wake_std = calculate_std(wake, wake_mean, "wake")
    restless_std = calculate_std(restless, restless_mean, "restless")
    light_std = calculate_std(light, light_mean, "light")
    deep_std = calculate_std(deep, deep_mean, "deep")
    rem_std = calculate_std(rem, rem_mean, "REM")
    sleep_std = calculate_std(sleep, sleep_mean, "sleep")

    # Variance
    wake_var = calculate_variance(wake_std, "wake")
    restless_var = calculate_variance(restless_std, "restless")
    light_var = calculate_variance(light_std, "light")
    deep_var = calculate_variance(deep_std, "deep")
    rem_var = calculate_variance(rem_std, "REM")
    sleep_var = calculate_variance(sleep_std, "sleep")

    # Differential entropy
    wake_entropy = calculate_differential_entropy(wake_lfps, "wake")
    restless_entropy = calculate_differential_entropy(restless_lfps, "restless")
    light_entropy = calculate_differential_entropy(light_lfps, "light")
    deep_entropy = calculate_differential_entropy(deep_lfps, "deep")
    rem_entropy = calculate_differential_entropy(rem_lfps, "REM")
    sleep_entropy = calculate_differential_entropy(sleep_lfps, "sleep")

    # Get the stage's LFP frequency
    wake_lfps = extract_lfps(wake)
    restless_lfps = extract_lfps(restless)
    light_lfps = extract_lfps(light)
    deep_lfps = extract_lfps(deep)
    rem_lfps = extract_lfps(rem)
    sleep_lfps = extract_lfps(sleep)

    # Compare each stage mean with the wake mean
    restless_comparison = compare_means(restless_mean, wake_mean, "restless")
    light_comparison = compare_means(light_mean, wake_mean, "light")
    deep_comparison = compare_means(deep_mean, wake_mean, "deep")
    rem_comparison = compare_means(rem_mean, wake_mean, "REM")
    sleep_comparison = compare_means(sleep_mean, wake_mean, "sleep")

    # Difference between each stage mean and the wake mean
    restless_difference = calculate_difference(restless_mean, wake_mean, "restless")
    light_difference = calculate_difference(light_mean, wake_mean, "light")
    deep_difference = calculate_difference(deep_mean, wake_mean, "deep")
    rem_difference = calculate_difference(rem_mean, wake_mean, "REM")
    sleep_difference = calculate_difference(sleep_mean, wake_mean, "sleep")

    # Write all results to new spreadsheet summarizing each stage's statistics
    with open(current_csv[0:3] + frequency_band + '_statistics.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Stage', 'Stage Number', 'Sleep or Wake', 'Mean', 'Wake Comparison', 'Wake Difference', 'Standard Deviation', 'Variance', 'Entropy'])
        writer.writerow(['Wake', 0, 0, wake_mean, 'N/A', 'N/A', wake_std, wake_var, wake_entropy])
        writer.writerow(['Restless', 1, 1, restless_mean, restless_comparison, restless_difference, restless_std, restless_var, restless_entropy])
        writer.writerow(['Light', 2, 1, light_mean, light_comparison, light_difference, light_std, light_var, light_entropy])
        writer.writerow(['Deep', 3, 1, deep_mean, deep_comparison, deep_difference, deep_std, deep_var, deep_entropy])
        writer.writerow(['REM', 4, 1, rem_mean, rem_comparison, rem_difference, rem_std, rem_var, rem_entropy])
        writer.writerow(['Sleep', 5, 1, sleep_mean, sleep_comparison, sleep_difference, sleep_std, sleep_var, sleep_entropy])

    complete_list.append([subject_n, frequency_band, 'Wake', 0, 0, wake_mean, 'N/A', 'N/A', wake_std, wake_var, wake_entropy])
    complete_list.append([subject_n, frequency_band, 'Restless', 1, 1, restless_mean, restless_comparison, restless_difference, restless_std, restless_var, restless_entropy])
    complete_list.append([subject_n, frequency_band, 'Light', 2, 1, light_mean, light_comparison, light_difference, light_std, light_var, light_entropy])
    complete_list.append([subject_n, frequency_band, 'Deep', 3, 1, deep_mean, deep_comparison, deep_difference, deep_std, deep_var, deep_entropy])
    complete_list.append([subject_n, frequency_band, 'REM', 4, 1, rem_mean, rem_comparison, rem_difference, rem_std, rem_var, rem_entropy])
    complete_list.append([subject_n, frequency_band, 'Sleep', 5, 1, sleep_mean, sleep_comparison, sleep_difference, sleep_std, sleep_var, sleep_entropy])

    # Write spreadsheets for each diagnosis, target, and frequency group
    if frequency_band == 'alpha':
        # Parkinson's disease group which had alpha recordings
        if subject_n == 2 or subject_n == 4 or subject_n == 5 or subject_n == 7 or subject_n == 10:
            allPD_alpha.append([subject_n, frequency_band, 'Wake', 0, 0, wake_mean, 'N/A', 'N/A', wake_std, wake_var, wake_entropy])
            allPD_alpha.append([subject_n, frequency_band, 'Restless', 1, 1, restless_mean, restless_comparison, restless_difference, restless_std, restless_var, restless_entropy])
            allPD_alpha.append([subject_n, frequency_band, 'Light', 2, 1, light_mean, light_comparison, light_difference, light_std, light_var, light_entropy])
            allPD_alpha.append([subject_n, frequency_band, 'Deep', 3, 1, deep_mean, deep_comparison, deep_difference, deep_std, deep_var, deep_entropy])
            allPD_alpha.append([subject_n, frequency_band, 'REM', 4, 1, rem_mean, rem_comparison, rem_difference, rem_std, rem_var, rem_entropy])
            allPD_alpha.append([subject_n, frequency_band, 'Sleep', 5, 1, sleep_mean, sleep_comparison, sleep_difference, sleep_std, sleep_var, sleep_entropy])
            # Specifically with subthalamic nucleus target
            if subject_n == 2 or subject_n == 5 or subject_n == 10:
                allPD_STN_alpha.append([subject_n, frequency_band, 'Wake', 0, 0, wake_mean, 'N/A', 'N/A', wake_std, wake_var, wake_entropy])
                allPD_STN_alpha.append([subject_n, frequency_band, 'Restless', 1, 1, restless_mean, restless_comparison, restless_difference, restless_std, restless_var, restless_entropy])
                allPD_STN_alpha.append([subject_n, frequency_band, 'Light', 2, 1, light_mean, light_comparison, light_difference, light_std, light_var, light_entropy])
                allPD_STN_alpha.append([subject_n, frequency_band, 'Deep', 3, 1, deep_mean, deep_comparison, deep_difference, deep_std, deep_var, deep_entropy])
                allPD_STN_alpha.append([subject_n, frequency_band, 'REM', 4, 1, rem_mean, rem_comparison, rem_difference, rem_std, rem_var, rem_entropy])
                allPD_STN_alpha.append([subject_n, frequency_band, 'Sleep', 5, 1, sleep_mean, sleep_comparison, sleep_difference, sleep_std, sleep_var, sleep_entropy])
            # Specifically with globus pallidus internus target
            elif subject_n == 4 or subject_n == 7:
                allPD_GPI_alpha.append([subject_n, frequency_band, 'Wake', 0, 0, wake_mean, 'N/A', 'N/A', wake_std, wake_var, wake_entropy])
                allPD_GPI_alpha.append([subject_n, frequency_band, 'Restless', 1, 1, restless_mean, restless_comparison, restless_difference, restless_std, restless_var, restless_entropy])
                allPD_GPI_alpha.append([subject_n, frequency_band, 'Light', 2, 1, light_mean, light_comparison, light_difference, light_std, light_var, light_entropy])
                allPD_GPI_alpha.append([subject_n, frequency_band, 'Deep', 3, 1, deep_mean, deep_comparison, deep_difference, deep_std, deep_var, deep_entropy])
                allPD_GPI_alpha.append([subject_n, frequency_band, 'REM', 4, 1, rem_mean, rem_comparison, rem_difference, rem_std, rem_var, rem_entropy])
                allPD_GPI_alpha.append([subject_n, frequency_band, 'Sleep', 5, 1, sleep_mean, sleep_comparison, sleep_difference, sleep_std, sleep_var, sleep_entropy])
        # Non-Parkinson's disease group which had alpha recordings
        elif subject_n == 6 or subject_n == 8 or subject_n == 9:
            allNonPD_alpha.append([subject_n, frequency_band, 'Wake', 0, 0, wake_mean, 'N/A', 'N/A', wake_std, wake_var, wake_entropy])
            allNonPD_alpha.append([subject_n, frequency_band, 'Restless', 1, 1, restless_mean, restless_comparison, restless_difference, restless_std, restless_var, restless_entropy])
            allNonPD_alpha.append([subject_n, frequency_band, 'Light', 2, 1, light_mean, light_comparison, light_difference, light_std, light_var, light_entropy])
            allNonPD_alpha.append([subject_n, frequency_band, 'Deep', 3, 1, deep_mean, deep_comparison, deep_difference, deep_std, deep_var, deep_entropy])
            allNonPD_alpha.append([subject_n, frequency_band, 'REM', 4, 1, rem_mean, rem_comparison, rem_difference, rem_std, rem_var, rem_entropy])
            allNonPD_alpha.append([subject_n, frequency_band, 'Sleep', 5, 1, sleep_mean, sleep_comparison, sleep_difference, sleep_std, sleep_var, sleep_entropy])
    else:
        # Parkinson's disease group which had beta recordings
        if subject_n == 2 or subject_n == 3 or subject_n == 4 or subject_n == 7 or subject_n == 10:
            allPD_beta.append([subject_n, frequency_band, 'Wake', 0, 0, wake_mean, 'N/A', 'N/A', wake_std, wake_var, wake_entropy])
            allPD_beta.append([subject_n, frequency_band, 'Restless', 1, 1, restless_mean, restless_comparison, restless_difference, restless_std, restless_var, restless_entropy])
            allPD_beta.append([subject_n, frequency_band, 'Light', 2, 1, light_mean, light_comparison, light_difference, light_std, light_var, light_entropy])
            allPD_beta.append([subject_n, frequency_band, 'Deep', 3, 1, deep_mean, deep_comparison, deep_difference, deep_std, deep_var, deep_entropy])
            allPD_beta.append([subject_n, frequency_band, 'REM', 4, 1, rem_mean, rem_comparison, rem_difference, rem_std, rem_var, rem_entropy])
            allPD_beta.append([subject_n, frequency_band, 'Sleep', 5, 1, sleep_mean, sleep_comparison, sleep_difference, sleep_std, sleep_var, sleep_entropy])
            # Specifically with subthalamic nucleus target
            if subject_n == 2 or subject_n == 3 or subject_n == 10:
                allPD_STN_beta.append([subject_n, frequency_band, 'Wake', 0, 0, wake_mean, 'N/A', 'N/A', wake_std, wake_var, wake_entropy])
                allPD_STN_beta.append([subject_n, frequency_band, 'Restless', 1, 1, restless_mean, restless_comparison, restless_difference, restless_std, restless_var, restless_entropy])
                allPD_STN_beta.append([subject_n, frequency_band, 'Light', 2, 1, light_mean, light_comparison, light_difference, light_std, light_var, light_entropy])
                allPD_STN_beta.append([subject_n, frequency_band, 'Deep', 3, 1, deep_mean, deep_comparison, deep_difference, deep_std, deep_var, deep_entropy])
                allPD_STN_beta.append([subject_n, frequency_band, 'REM', 4, 1, rem_mean, rem_comparison, rem_difference, rem_std, rem_var, rem_entropy])
                allPD_STN_beta.append([subject_n, frequency_band, 'Sleep', 5, 1, sleep_mean, sleep_comparison, sleep_difference, sleep_std, sleep_var, sleep_entropy])
            # Specifically with globus pallidus internus target
            elif subject_n == 4 or subject_n == 7:
                allPD_GPI_beta.append([subject_n, frequency_band, 'Wake', 0, 0, wake_mean, 'N/A', 'N/A', wake_std, wake_var, wake_entropy])
                allPD_GPI_beta.append([subject_n, frequency_band, 'Restless', 1, 1, restless_mean, restless_comparison, restless_difference, restless_std, restless_var, restless_entropy])
                allPD_GPI_beta.append([subject_n, frequency_band, 'Light', 2, 1, light_mean, light_comparison, light_difference, light_std, light_var, light_entropy])
                allPD_GPI_beta.append([subject_n, frequency_band, 'Deep', 3, 1, deep_mean, deep_comparison, deep_difference, deep_std, deep_var, deep_entropy])
                allPD_GPI_beta.append([subject_n, frequency_band, 'REM', 4, 1, rem_mean, rem_comparison, rem_difference, rem_std, rem_var, rem_entropy])
                allPD_GPI_beta.append([subject_n, frequency_band, 'Sleep', 5, 1, sleep_mean, sleep_comparison, sleep_difference, sleep_std, sleep_var, sleep_entropy])
    # Essential tremor group which only had alpha recoridngs
    if subject_n == 8 or subject_n == 9:
        allET_alpha.append([subject_n, frequency_band, 'Wake', 0, 0, wake_mean, 'N/A', 'N/A', wake_std, wake_var, wake_entropy])
        allET_alpha.append([subject_n, frequency_band, 'Restless', 1, 1, restless_mean, restless_comparison, restless_difference, restless_std, restless_var, restless_entropy])
        allET_alpha.append([subject_n, frequency_band, 'Light', 2, 1, light_mean, light_comparison, light_difference, light_std, light_var, light_entropy])
        allET_alpha.append([subject_n, frequency_band, 'Deep', 3, 1, deep_mean, deep_comparison, deep_difference, deep_std, deep_var, deep_entropy])
        allET_alpha.append([subject_n, frequency_band, 'REM', 4, 1, rem_mean, rem_comparison, rem_difference, rem_std, rem_var, rem_entropy])
        allET_alpha.append([subject_n, frequency_band, 'Sleep', 5, 1, sleep_mean, sleep_comparison, sleep_difference, sleep_std, sleep_var, sleep_entropy])

# Writes the statistics for a disease group and frequency to a spreadsheet
def write_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Subject Number', 'Frequency Band', 'Stage', 'Stage Number', 'Sleep or Wake', 'Mean', 'Wake Comparison', 'Wake Difference', 'Standard Deviation', 'Variance', 'Entropy'])
        writer.writerows(data)

write_to_csv('statistics.csv', complete_list)
write_to_csv('allPD_alpha.csv', allPD_alpha)
write_to_csv('allPD_beta.csv', allPD_beta)
write_to_csv('allPD_STN_alpha.csv', allPD_STN_alpha)
write_to_csv('allPD_GPI_alpha.csv', allPD_GPI_alpha)
write_to_csv('allPD_STN_beta.csv', allPD_STN_beta)
write_to_csv('allPD_GPI_beta.csv', allPD_GPI_beta)
write_to_csv('allNonPD_alpha.csv', allNonPD_alpha)
write_to_csv('allET_alpha.csv', allET_alpha)
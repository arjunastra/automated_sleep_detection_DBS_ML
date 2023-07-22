# statistics.py calculates statistical values for groups and subgroups based on their diagnosis and target
# statistics are written to .csv files and can be used for further analysis, Author: yhashim 
import csv
import math

from scipy.stats import differential_entropy

file_names = ['002left_LFP', '002right_LFP', '003right_LFP', '004left_LFP', '004right_LFP', '005right_LFP', '006left_LFP', '006right_LFP', '007left_LFP', '007right_LFP', '008left_LFP', '009left_LFP', '010left_LFP', '010right_LFP']

complete_list = []

allPD_alpha, allPD_beta = [], []

allPD_STN_beta, allPD_STN_alpha = [], []
allPD_GPI_beta, allPD_GPI_alpha = [], []

allNonPD_alpha = []

allNonPD_STN_alpha, allNonPD_GPI_alpha = [], []

allET_alpha = []

def calculate_mean(data):
    try:
        return sum([float(row[1]) for row in data]) / len(data)
    except Exception as e:
        print("mean error raised is:", e)
        return 'NULL'

def calculate_std(data, mean):
    try:
        return (sum([(float(row[1]) - mean)**2 for row in data]) / len(data)) ** 0.5
    except Exception as e:
        print("standard deviation error raised is:", e)
        return 'NULL'

def calculate_var(std):
    try:
        return std ** 2
    except Exception as e:
        print("variance error raised is:", e)
        return 'NULL'
    
def calculate_stage_data(data, stage_name, stage_number, sleep_or_wake, mean, std, var, entropy):
    stage_lfps = [float(row[1]) for row in data]
    try:
        entropy_value = differential_entropy(stage_lfps)
    except Exception as e:
        print(f"{stage_name} differential entropy error raised is:", e)
        entropy_value = 'NULL'
    
    return [stage_name, stage_number, sleep_or_wake, mean, 'N/A', 'N/A', std, var, entropy_value]

def get_comparison(mean, wake_mean):
    try:
        if mean > wake_mean:
            return 'greater'
        elif mean < wake_mean:
            return 'less'
        else:
            return 'equal'
    except Exception as e:
        print("error raised is: ", e)
        return 'N/A'

def calculate_difference(stage_mean, wake_mean):
    try:
        return stage_mean - wake_mean
    except Exception as e:
        print("error raised is: ", e)
        return 'NULL'


def write_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Subject Number', 'Frequency Band', 'Stage', 'Stage Number', 'Sleep or Wake', 'Mean', 'Wake Comparison', 'Wake Difference', 'Standard Deviation', 'Variance', 'Entropy'])
        writer.writerows(data)


for file_name in file_names:
    current_csv = file_name + '.csv'
    subject_n = int(current_csv[0:3])

    # read csv file and store in list
    # skip first row
    with open(current_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        data = list(reader)

    frequency_band = data[0][5]

    wake, restless, light, deep, rem = [], [], [], [], []
    sleepAdd = []

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

    wake_mean = restless_mean = light_mean = deep_mean = rem_mean = sleep_mean = 'NULL'
    wake_std = restless_std = light_std = deep_std = rem_std = sleep_std = 'NULL'
    wake_var = restless_var = light_var = deep_var = rem_var = sleep_var = 'NULL'
    wake_entropy = restless_entropy = light_entropy = deep_entropy = rem_entropy = sleep_entropy = 'NULL'

    wake_mean = calculate_mean(wake)
    restless_mean = calculate_mean(restless)
    light_mean = calculate_mean(light)
    deep_mean = calculate_mean(deep)
    rem_mean = calculate_mean(rem)
    sleep_mean = calculate_mean(sleep)

    wake_std = calculate_std(wake, wake_mean)
    restless_std = calculate_std(restless, restless_mean)
    light_std = calculate_std(light, light_mean)
    deep_std = calculate_std(deep, deep_mean)
    rem_std = calculate_std(rem, rem_mean)
    sleep_std = calculate_std(sleep, sleep_mean)

    wake_var = calculate_var(wake_std)
    restless_var = calculate_var(restless_std)
    light_var = calculate_var(light_std)
    deep_var = calculate_var(deep_std)
    rem_var = calculate_var(rem_std)
    sleep_var = calculate_var(sleep_std)    

    wake_lfps, restless_lfps, light_lfps, deep_lfps, rem_lfps, sleep_lfps = [], [], [], [], [], []
    for row in wake:
        wake_lfps.append(float(row[1]))
    for row in restless:
        restless_lfps.append(float(row[1]))
    for row in light:
        light_lfps.append(float(row[1]))
    for row in deep:
        deep_lfps.append(float(row[1]))
    for row in rem:
        rem_lfps.append(float(row[1]))
    for row in sleep:
        sleep_lfps.append(float(row[1]))

    restrestless_comparison = get_comparison(restless_mean, wake_mean)
    light_comparison = get_comparison(light_mean, wake_mean)
    deep_comparison = get_comparison(deep_mean, wake_mean)
    rem_comparison = get_comparison(rem_mean, wake_mean)
    sleep_comparison = get_comparison(sleep_mean, wake_mean)

    restless_difference = calculate_difference(restless_mean, wake_mean)
    light_difference = calculate_difference(light_mean, wake_mean)
    deep_difference = calculate_difference(deep_mean, wake_mean)
    rem_difference = calculate_difference(rem_mean, wake_mean)
    sleep_difference = calculate_difference(sleep_mean, wake_mean)

    # write results to new csv file summarizing each stage's statistics in individual rows
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

    # beta:
    # pd: 002left, 003right, 004left, 007right, 010right
    # non-pd: 006right (no)

    # alpha: 
    # pd: 002right, 004right, 005right, 007left, 010left
    # non-pd: 006left, 008left, 009left

    # pd: 
    # stn: 002, 003, 005, 010
    # gpi: 004, 007

    # nonpd: 
    # gpi: 006
    # vim: 008, 009 

    if frequency_band == 'alpha':
        if subject_n == 2 or subject_n == 4 or subject_n == 5 or subject_n == 7 or subject_n == 10:
            allPD_alpha.append([subject_n, frequency_band, 'Wake', 0, 0, wake_mean, 'N/A', 'N/A', wake_std, wake_var, wake_entropy])
            allPD_alpha.append([subject_n, frequency_band, 'Restless', 1, 1, restless_mean, restless_comparison, restless_difference, restless_std, restless_var, restless_entropy])
            allPD_alpha.append([subject_n, frequency_band, 'Light', 2, 1, light_mean, light_comparison, light_difference, light_std, light_var, light_entropy])
            allPD_alpha.append([subject_n, frequency_band, 'Deep', 3, 1, deep_mean, deep_comparison, deep_difference, deep_std, deep_var, deep_entropy])
            allPD_alpha.append([subject_n, frequency_band, 'REM', 4, 1, rem_mean, rem_comparison, rem_difference, rem_std, rem_var, rem_entropy])
            allPD_alpha.append([subject_n, frequency_band, 'Sleep', 5, 1, sleep_mean, sleep_comparison, sleep_difference, sleep_std, sleep_var, sleep_entropy])
            if subject_n == 2 or subject_n == 5 or subject_n == 10:
                allPD_STN_alpha.append([subject_n, frequency_band, 'Wake', 0, 0, wake_mean, 'N/A', 'N/A', wake_std, wake_var, wake_entropy])
                allPD_STN_alpha.append([subject_n, frequency_band, 'Restless', 1, 1, restless_mean, restless_comparison, restless_difference, restless_std, restless_var, restless_entropy])
                allPD_STN_alpha.append([subject_n, frequency_band, 'Light', 2, 1, light_mean, light_comparison, light_difference, light_std, light_var, light_entropy])
                allPD_STN_alpha.append([subject_n, frequency_band, 'Deep', 3, 1, deep_mean, deep_comparison, deep_difference, deep_std, deep_var, deep_entropy])
                allPD_STN_alpha.append([subject_n, frequency_band, 'REM', 4, 1, rem_mean, rem_comparison, rem_difference, rem_std, rem_var, rem_entropy])
                allPD_STN_alpha.append([subject_n, frequency_band, 'Sleep', 5, 1, sleep_mean, sleep_comparison, sleep_difference, sleep_std, sleep_var, sleep_entropy])
            elif subject_n == 4 or subject_n == 7:
                allPD_GPI_alpha.append([subject_n, frequency_band, 'Wake', 0, 0, wake_mean, 'N/A', 'N/A', wake_std, wake_var, wake_entropy])
                allPD_GPI_alpha.append([subject_n, frequency_band, 'Restless', 1, 1, restless_mean, restless_comparison, restless_difference, restless_std, restless_var, restless_entropy])
                allPD_GPI_alpha.append([subject_n, frequency_band, 'Light', 2, 1, light_mean, light_comparison, light_difference, light_std, light_var, light_entropy])
                allPD_GPI_alpha.append([subject_n, frequency_band, 'Deep', 3, 1, deep_mean, deep_comparison, deep_difference, deep_std, deep_var, deep_entropy])
                allPD_GPI_alpha.append([subject_n, frequency_band, 'REM', 4, 1, rem_mean, rem_comparison, rem_difference, rem_std, rem_var, rem_entropy])
                allPD_GPI_alpha.append([subject_n, frequency_band, 'Sleep', 5, 1, sleep_mean, sleep_comparison, sleep_difference, sleep_std, sleep_var, sleep_entropy])
        elif subject_n == 6 or subject_n == 8 or subject_n == 9:
            allNonPD_alpha.append([subject_n, frequency_band, 'Wake', 0, 0, wake_mean, 'N/A', 'N/A', wake_std, wake_var, wake_entropy])
            allNonPD_alpha.append([subject_n, frequency_band, 'Restless', 1, 1, restless_mean, restless_comparison, restless_difference, restless_std, restless_var, restless_entropy])
            allNonPD_alpha.append([subject_n, frequency_band, 'Light', 2, 1, light_mean, light_comparison, light_difference, light_std, light_var, light_entropy])
            allNonPD_alpha.append([subject_n, frequency_band, 'Deep', 3, 1, deep_mean, deep_comparison, deep_difference, deep_std, deep_var, deep_entropy])
            allNonPD_alpha.append([subject_n, frequency_band, 'REM', 4, 1, rem_mean, rem_comparison, rem_difference, rem_std, rem_var, rem_entropy])
            allNonPD_alpha.append([subject_n, frequency_band, 'Sleep', 5, 1, sleep_mean, sleep_comparison, sleep_difference, sleep_std, sleep_var, sleep_entropy])
    else:
        if subject_n == 2 or subject_n == 3 or subject_n == 4 or subject_n == 7 or subject_n == 10:
            allPD_beta.append([subject_n, frequency_band, 'Wake', 0, 0, wake_mean, 'N/A', 'N/A', wake_std, wake_var, wake_entropy])
            allPD_beta.append([subject_n, frequency_band, 'Restless', 1, 1, restless_mean, restless_comparison, restless_difference, restless_std, restless_var, restless_entropy])
            allPD_beta.append([subject_n, frequency_band, 'Light', 2, 1, light_mean, light_comparison, light_difference, light_std, light_var, light_entropy])
            allPD_beta.append([subject_n, frequency_band, 'Deep', 3, 1, deep_mean, deep_comparison, deep_difference, deep_std, deep_var, deep_entropy])
            allPD_beta.append([subject_n, frequency_band, 'REM', 4, 1, rem_mean, rem_comparison, rem_difference, rem_std, rem_var, rem_entropy])
            allPD_beta.append([subject_n, frequency_band, 'Sleep', 5, 1, sleep_mean, sleep_comparison, sleep_difference, sleep_std, sleep_var, sleep_entropy])
            if subject_n == 2 or subject_n == 3 or subject_n == 10:
                allPD_STN_beta.append([subject_n, frequency_band, 'Wake', 0, 0, wake_mean, 'N/A', 'N/A', wake_std, wake_var, wake_entropy])
                allPD_STN_beta.append([subject_n, frequency_band, 'Restless', 1, 1, restless_mean, restless_comparison, restless_difference, restless_std, restless_var, restless_entropy])
                allPD_STN_beta.append([subject_n, frequency_band, 'Light', 2, 1, light_mean, light_comparison, light_difference, light_std, light_var, light_entropy])
                allPD_STN_beta.append([subject_n, frequency_band, 'Deep', 3, 1, deep_mean, deep_comparison, deep_difference, deep_std, deep_var, deep_entropy])
                allPD_STN_beta.append([subject_n, frequency_band, 'REM', 4, 1, rem_mean, rem_comparison, rem_difference, rem_std, rem_var, rem_entropy])
                allPD_STN_beta.append([subject_n, frequency_band, 'Sleep', 5, 1, sleep_mean, sleep_comparison, sleep_difference, sleep_std, sleep_var, sleep_entropy])
            elif subject_n == 4 or subject_n == 7:
                allPD_GPI_beta.append([subject_n, frequency_band, 'Wake', 0, 0, wake_mean, 'N/A', 'N/A', wake_std, wake_var, wake_entropy])
                allPD_GPI_beta.append([subject_n, frequency_band, 'Restless', 1, 1, restless_mean, restless_comparison, restless_difference, restless_std, restless_var, restless_entropy])
                allPD_GPI_beta.append([subject_n, frequency_band, 'Light', 2, 1, light_mean, light_comparison, light_difference, light_std, light_var, light_entropy])
                allPD_GPI_beta.append([subject_n, frequency_band, 'Deep', 3, 1, deep_mean, deep_comparison, deep_difference, deep_std, deep_var, deep_entropy])
                allPD_GPI_beta.append([subject_n, frequency_band, 'REM', 4, 1, rem_mean, rem_comparison, rem_difference, rem_std, rem_var, rem_entropy])
                allPD_GPI_beta.append([subject_n, frequency_band, 'Sleep', 5, 1, sleep_mean, sleep_comparison, sleep_difference, sleep_std, sleep_var, sleep_entropy])
    
    if subject_n == 8 or subject_n == 9:
        allET_alpha.append([subject_n, frequency_band, 'Wake', 0, 0, wake_mean, 'N/A', 'N/A', wake_std, wake_var, wake_entropy])
        allET_alpha.append([subject_n, frequency_band, 'Restless', 1, 1, restless_mean, restless_comparison, restless_difference, restless_std, restless_var, restless_entropy])
        allET_alpha.append([subject_n, frequency_band, 'Light', 2, 1, light_mean, light_comparison, light_difference, light_std, light_var, light_entropy])
        allET_alpha.append([subject_n, frequency_band, 'Deep', 3, 1, deep_mean, deep_comparison, deep_difference, deep_std, deep_var, deep_entropy])
        allET_alpha.append([subject_n, frequency_band, 'REM', 4, 1, rem_mean, rem_comparison, rem_difference, rem_std, rem_var, rem_entropy])
        allET_alpha.append([subject_n, frequency_band, 'Sleep', 5, 1, sleep_mean, sleep_comparison, sleep_difference, sleep_std, sleep_var, sleep_entropy])

write_to_csv('statistics.csv', complete_list)
write_to_csv('allPD_alpha.csv', allPD_alpha)
write_to_csv('allPD_beta.csv', allPD_beta)
write_to_csv('allPD_STN_alpha.csv', allPD_STN_alpha)
write_to_csv('allPD_GPI_alpha.csv', allPD_GPI_alpha)
write_to_csv('allPD_STN_beta.csv', allPD_STN_beta)
write_to_csv('allPD_GPI_beta.csv', allPD_GPI_beta)
write_to_csv('allNonPD_alpha.csv', allNonPD_alpha)
write_to_csv('allET_alpha.csv', allET_alpha)

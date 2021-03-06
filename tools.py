import pandas as pd
import re
import os
import os.path
import json
import numpy as np


def event_identifier():
    with open("data/social_assements_30_annotations.tsv", 'r') as data_file:
        df = pd.read_csv(data_file, sep='\t')
        events = df['event_identifier']
        uniq_events = set(events)
        print(len(events) - len(uniq_events))


def pattern_and_verbalizer_getter_writer():
    with open('patterns.py', 'r') as pattern_file:
        with open('verbalizers.py', 'r') as verbalizer_file:
            pattern_lines = pattern_file.readlines()
            verbalizer_lines = verbalizer_file.readlines()

            pattern_function_str = ''
            verbalizer_function_str = ''

            for line in pattern_lines:
                if 'def ' in line and re.search(r'_pattern_\d', line):
                    func_name = line[4:].split('(')[0]
                    pattern_function_str += '    elif pattern_string == \'{0}\':\n        return {0}\n'.format(func_name)

            for line in verbalizer_lines:
                if 'class ' in line and re.search(r'_verbalizer_\d', line):
                    func_name = line[6:].split('(')[0]
                    verbalizer_function_str += '    elif verbalizer_string == \'{0}\':\n        return {0}(args)\n'.format(func_name)

            string = pattern_function_str + '\n\n' + verbalizer_function_str
            with open('tools_dump_file.txt', 'w') as file:
                file.write(string)


def results_getter():
    files = []
    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f == 'log.json']:
            files.append(os.path.join(dirpath, filename))

    results = []
    for file in files:
        with open(file, 'r') as json_file:
            json_data = json.load(json_file)
            if 'eval_accuracy' in json_data:
                results.append(json_data['eval_accuracy'][-1])

    for file, result in zip(files, results):
        print(file + ': ' + str(result))

    print('average score is: ' + str(np.mean(results)))


# pattern_and_verbalizer_getter_writer()
results_getter()

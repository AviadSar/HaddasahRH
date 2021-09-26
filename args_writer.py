import os
import json


args_file_dir = "args"
data_file = "data/social_assesments_100_annotations_clean_filled_en.tsv"
target_names = ["marital_status"]
model_name = "roberta-base"
model_type = "MLM"

n_train_samples = 30
n_dev_samples = 30
n_test_samples = 30

num_patterns_list = [2]
patterns = ["marital_status_pattern_1", "marital_status_pattern_2"]
verbalizers = ["marital_status_verbalizer_1", "marital_status_verbalizer_2"]

num_labels = 2
labels_list = [["married", "not_married"]]
label_dictionary_list = [[["unknown", "not_married"], ["single", "not_married"], ["divorced", "not_married"], ["widowed", "not_married"]]]

pattern_batch_size = 1
pattern_logging_steps = 15
pattern_eval_steps = 60
pattern_gradient_accumulation_steps = 3
pattern_warmup_steps = 15
pattern_num_evals = 1
pattern_dropout = 0.1

classifier_batch_size = 2
classifier_logging_steps = 10
classifier_eval_steps = 1
classifier_gradient_accumulation_steps = 2
classifier_warmup_steps = 150
classifier_num_evals = 10
classifier_dropout = 0.1


for target_name, num_patterns, labels, label_dictionary in zip(target_names, num_patterns_list, labels_list, label_dictionary_list):

    args_file_name = "/" + target_name + ".json"
    pattern_model_dir = "model_outputs/" + target_name + "_pattern"
    classifier_model_dir = "model_outputs/" + target_name
    patterns = [target_name + "_pattern_" + str(idx) for idx in range(int(num_patterns))]
    verbalizers = [target_name + "_verbalizer_" + str(idx) for idx in range(int(num_patterns))]

    data_dict = {
                  "data_file": data_file,
                  "pattern_model_dir": pattern_model_dir,
                  "classifier_model_dir": classifier_model_dir,
                  "model_name": model_name,
                  "model_type": model_type,
                  "n_train_samples": n_train_samples,
                  "n_dev_samples": n_dev_samples,
                  "n_test_samples": n_test_samples,
                  "target_column": target_name,
                  "patterns": patterns,
                  "verbalizers": verbalizers,
                  "num_labels": num_labels,
                  "labels": labels,
                  "label_dictionary": label_dictionary,
                  "pattern_batch_size": pattern_batch_size,
                  "pattern_logging_steps": pattern_logging_steps,
                  "pattern_eval_steps": pattern_eval_steps,
                  "pattern_gradient_accumulation_steps": pattern_gradient_accumulation_steps,
                  "pattern_warmup_steps": pattern_warmup_steps,
                  "pattern_num_evals": pattern_num_evals,
                  "pattern_dropout": pattern_dropout,
                  "classifier_batch_size": classifier_batch_size,
                  "classifier_logging_steps": classifier_logging_steps,
                  "classifier_eval_steps": classifier_eval_steps,
                  "classifier_gradient_accumulation_steps": classifier_gradient_accumulation_steps,
                  "classifier_warmup_steps": classifier_warmup_steps,
                  "classifier_num_evals": classifier_num_evals,
                  "classifier_dropout": classifier_dropout,
                }

    if not os.path.isdir(args_file_dir):
        os.makedirs(args_file_dir)
        print("creating dataset directory " + args_file_dir)
    with open(args_file_dir + args_file_name, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)
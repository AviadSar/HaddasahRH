import os
import json


args_file_dir = "args"
data_file = "data/social_assesments_100_annotations_clean_filled_en.tsv"
target_names = ["sex",
                "immigrant",
                "marital_status",
                "children",
                "closest_relative",
                "closest_supporting_relative",
                "help_at_home_hours",
                "seeking_help_at_home",
                "is_exhausted",
                "needs_extreme_nursing",
                "has_extreme_nursing",
                "is_confused",
                "is_dementic",
                "residence",
                "recommended_residence"]

model_name = "roberta-base"
model_type = "MLM"

n_train_samples = 30
n_dev_samples = 30
n_test_samples = 30

num_patterns_list = [4, #sex
                     4, #immigrant
                     3, #marital_status
                     2, #children
                     2, #closest_relative
                     2, #closest_supporting_relative
                     3, #help_at_home_hours
                     3, #seeking_help_at_home
                     3, #is_exhausted
                     2, #needs_extreme_nursing
                     2, #has_extreme_nursing
                     3, #is_confused
                     3, #is_dementic
                     3, #residence
                     3]

num_labels_list = [3, #sex
                   2, #immigrant
                   2, #marital_status
                   3, #children
                   3, #closest_relative
                   3, #closest_supporting_relative
                   4, #help_at_home_hours
                   2, #seeking_help_at_home
                   2, #is_exhausted
                   2, #needs_extreme_nursing
                   2, #has_extreme_nursing
                   2, #is_confused
                   2, #is_dementic
                   2, #residence
                   2]

labels_list = [["m", "f", "unknown"],
               ["yes", "no"],
               ["married", "not_married"],
               ["yes", "no", "unknown"],
               ["at_home", "close", "far"],
               ["at_home", "close", "far"],
               ["no", "few", "many", "special"],
               ["yes", "no"],
               ["yes", "no"],
               ["yes", "no"],
               ["yes", "no"],
               ["yes", "no"], #is_confused
               ["yes", "no"],
               ["home", "nursing_home"],
               ["home", "nursing_home"]]

label_dictionary_list = [[],
                         [],
                         [["unknown", "not_married"], ["single", "not_married"], ["divorced", "not_married"], ["widowed", "not_married"]],
                         [],
                         [["unknown", "far"]],
                         [["unknown", "far"]],
                         [],
                         [],
                         [],
                         [],
                         [],
                         [], #is_confused
                         [],
                         [["family_member_home", "home"]],
                         []]

pattern_batch_size = 6
pattern_logging_steps = 1
pattern_eval_steps = 1
pattern_gradient_accumulation_steps = 5
pattern_warmup_steps = 15
pattern_num_evals = 5
pattern_dropout = 0.1

classifier_batch_size = 16
classifier_logging_steps = 10
classifier_eval_steps = 150
classifier_gradient_accumulation_steps = 2
classifier_warmup_steps = 150
classifier_num_evals = 10
classifier_dropout = 0.1


for target_name, num_patterns, num_labels, labels, label_dictionary in zip(target_names, num_patterns_list, num_labels_list, labels_list, label_dictionary_list):

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
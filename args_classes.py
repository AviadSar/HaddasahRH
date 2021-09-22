import json


class Args(object):
    def __init__(self, json_file):
        with open(json_file, 'r') as json_file:
            json_data = json.load(json_file)
            self.data_file = json_data["data_file"]
            self.pattern_model_dir = json_data["pattern_model_dir"]
            self.classifier_model_dir = json_data["classifier_model_dir"]
            self.model_name = json_data["model_name"]
            self.model_type = json_data["model_type"]

            self.n_train_samples = json_data["n_train_samples"]
            self.n_dev_samples = json_data["n_dev_samples"]
            self.n_test_samples = json_data["n_test_samples"]

            self.target_column = json_data["target_column"]
            self.patterns = json_data["patterns"]
            self.verbalizers = json_data["verbalizers"]

            self.num_labels = json_data["num_labels"]
            self.labels = json_data["labels"]
            self.label_dictionary = json_data["label_dictionary"]

            self.pattern_batch_size = json_data["pattern_batch_size"]
            self.pattern_logging_steps = json_data["pattern_logging_steps"]
            self.pattern_eval_steps = json_data["pattern_eval_steps"]
            self.pattern_gradient_accumulation_steps = json_data["pattern_gradient_accumulation_steps"]
            self.pattern_warmup_steps = json_data["pattern_warmup_steps"]
            self.pattern_num_evals = json_data["pattern_num_evals"]
            self.pattern_dropout = json_data["pattern_dropout"]

            self.classifier_batch_size = json_data["classifier_batch_size"]
            self.classifier_logging_steps = json_data["classifier_logging_steps"]
            self.classifier_eval_steps = json_data["classifier_eval_steps"]
            self.classifier_gradient_accumulation_steps = json_data["classifier_gradient_accumulation_steps"]
            self.classifier_warmup_steps = json_data["classifier_warmup_steps"]
            self.classifier_num_evals = json_data["classifier_num_evals"]
            self.classifier_dropout = json_data["classifier_dropout"]

            if "eval" in json_data:
                self.eval = json_data["eval"]
            else:
                self.eval = False

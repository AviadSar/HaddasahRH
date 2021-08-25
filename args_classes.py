import json


class Args(object):
    def __init__(self, json_file):
        with open(json_file, 'r') as json_file:
            json_data = json.load(json_file)
            self.data_file = json_data["model_dir"]
            self.model_dir = json_data["data_file"]
            self.model_name = json_data["model_name"]
            self.model_type = json_data["model_type"]
            self.batch_size = json_data["batch_size"]

            self.n_train_samples = json_data["n_train_samples"]
            self.n_dev_samples = json_data["n_dev_samples"]
            self.n_test_samples = json_data["n_test_samples"]

            self.target_column = json_data["target_column"]
            self.patterns = json_data["patterns"]
            self.verbalizers = json_data["verbalizers"]

            self.num_labels = json_data["num_labels"]
            self.labels = json_data["labels"]
            self.label_dictionary = json_data["label_dictionary"]

            self.logging_steps = json_data["logging_steps"]
            self.eval_steps = json_data["eval_steps"]
            self.num_evals = json_data["num_evals"]
            self.dropout = json_data["dropout"]

# PlanCLEFTask
To Train: First run get_list_files_all to get train_set.pkl
To Validate: create a folder dirctory and move all model into it
- To validate a single model, use validate: validate.py <list_files> <model_path>
- To validate all models sequentially, use validate_all: validate_all <list_files> <model_directory>
- To validate all models with ensemble prediction, use validate_cross: validate_cross <train_set> <list_files> <model_directory>

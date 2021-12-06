
# --------------------------------------------------------------------------------------------
# FULL PIPELINE
# --------------------------------------------------------------------------------------------
full_pipeline: data_pipeline optimization_pipeline train_pipeline eval_pipeline

skip_data_pipeline: optimization_pipeline train_pipeline eval_pipeline

# --------------------------------------------------------------------------------------------
# PREPARE DATA
# --------------------------------------------------------------------------------------------
data_pipeline: clean_data train_test_split

clean_data:
	python prepare_data/clean_data.py

train_test_split:
	python prepare_data/train_test_split.py

# --------------------------------------------------------------------------------------------
# OPTIMIZATION
# --------------------------------------------------------------------------------------------
optimization_pipeline: optimization_sgd optimization_nb

optimization_sgd:
	python hyperparam_search/optuna_search_sgd.py

optimization_nb:
	python hyperparam_search/optuna_search_nb.py

# --------------------------------------------------------------------------------------------
# TRAIN
# --------------------------------------------------------------------------------------------
train_pipeline: train_sgd train_nb

train_sgd:
	python train/train_best_sgd.py

train_nb:
	python train/train_best_nb.py

# --------------------------------------------------------------------------------------------
# EVALUATION ON TEST
# --------------------------------------------------------------------------------------------
eval_pipeline: eval_best_sgd eval_best_nb

eval_best_sgd:
	python evaluation/sgd_evaluation.py

eval_best_nb:
	python evaluation/nb_evaluation.py

# --------------------------------------------------------------------------------------------
# TRAIN ON FULL DATASET
# --------------------------------------------------------------------------------------------
final_train_pipeline: final_train_sgd final_train_nb

final_train_sgd:
	python train/train_final_sgd.py

final_train_nb:
	python train/train_final_nb.py

# --------------------------------------------------------------------------------------------
# CHECK FEATURES
# --------------------------------------------------------------------------------------------
check_features:
	python evaluation/check_features.py

# --------------------------------------------------------------------------------------------
# DELETE OPTUNA STUDIES
# --------------------------------------------------------------------------------------------
delete_studies:
	python hyperparam_search/delete_studies.py
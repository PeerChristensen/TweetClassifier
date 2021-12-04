
# --------------------------------------------------------------------------------------------
# FULL PIPELINE
# --------------------------------------------------------------------------------------------
full_pipeline: prepare_data

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
# EVALUATION ON TEST
# --------------------------------------------------------------------------------------------
test_pipeline: test_best_sgd test_best_nb

test_best_sgd:
	python evaluation/test_best_sgd.py

test_best_nb:
	python evaluation/test_best_nb.py

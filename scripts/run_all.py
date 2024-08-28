import os
from pathlib import Path

import joblib
import yaml

from Reproduce_RuleGNN.src.get_datasets import get_real_world_datasets
from Reproduce_RuleGNN.src.preprocessing import preprocessing
from scripts.Evaluation.EvaluationFinal import model_selection_evaluation, best_model_evaluation
from scripts.find_best_models import find_best_models
from scripts.run_best_models import run_best_models


def run_all(database_names, cross_validations, config_files):
    # set omp_num_threads to 1 to avoid conflicts with OpenMP
    os.environ['OMP_NUM_THREADS'] = '1'
    # iterate over the databases
    for i, db_name in enumerate(database_names):


        cross_validation = cross_validations[i]
        config_file = config_files[i]
        # load the config file
        config = yaml.load(open(config_file), Loader=yaml.FullLoader)

        # find the best model hyperparameters using grid search and cross-validation
        joblib.Parallel(n_jobs=cross_validations[i])(joblib.delayed(find_best_models)(graph_db_name=db_name, validation_number=cross_validation, validation_id=id, graph_format="NEL", transfer=None, config=config_file) for id in range(cross_validations[i]))

        # run the best models
        # parallelize over (run_id, validation_id) pairs
        parallelization_pairs = [(run_id, validation_id) for run_id in range(3) for validation_id in
                                 range(cross_validation)]
        num_jobs = len(parallelization_pairs)
        joblib.Parallel(n_jobs=num_jobs)(joblib.delayed(run_best_models)(graph_db_name=db_name, run_id=run_id,
                                                                         validation_number=cross_validation,
                                                                         validation_id=validation_id,
                                                                         graph_format="NEL", transfer=None,
                                                                         config=config_file, evaluation_type='accuracy')
                                         for run_id, validation_id in parallelization_pairs)

        # evaluate the best models
        model_selection_evaluation(db_name, Path(config['paths']['results']), evaluation_type='accuracy')
        best_model_evaluation(db_name, Path(config['paths']['results']), evaluation_type='accuracy')

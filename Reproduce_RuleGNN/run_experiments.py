import joblib

from scripts.ModelSelection import main


def run():
    # use experiments.sh
    database_names = ['CSL', 'EvenOddRingsCount16', 'LongRings100', 'Snowflakes', 'NCI1', 'NCI109', 'Mutagenicity', 'DHFR', 'IMDB-BINARY', 'IMDB-MULTI']
    cross_validations = [5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    config_files = ['Reproduce_RuleGNN/Configs/config_CSL.yml', 'Reproduce_RuleGNN/Configs/config_EvenOddRings.yml', 'Reproduce_RuleGNN/Configs/config_EvenOddRingsCount.yml', 'Reproduce_RuleGNN/Configs/config_LongRings.yml', 'Reproduce_RuleGNN/Configs/config_Snowflakes.yml', 'Reproduce_RuleGNN/Configs/config_NCI1.yml', 'Reproduce_RuleGNN/Configs/config_NCI1.yml', 'Reproduce_RuleGNN/Configs/config_NCI1.yml', 'Reproduce_RuleGNN/Configs/config_DHFR.yml', 'Reproduce_RuleGNN/Configs/config_IMDB.yml', 'Reproduce_RuleGNN/Configs/config_IMDB.yml']
    # set omp_num_threads to 1 to avoid conflicts with OpenMP
    omp_num_threads = 1
    for i, db_name in enumerate(database_names):
        cross_validation = cross_validations[i]
        config_file = config_files[i]
        # use joblib to parallelize over the cross validations
        #joblib.Parallel(n_jobs=cross_validations[i])(joblib.delayed(ModelSelection.main)(graph_db_name=db_name, validation_number=cross_validation, validation_id=id, format="NEL", transfer=None, config=config_file) for id in range(cross_validations[i]))
        for id in range(cross_validation):
            main()
if __name__ == '__main__':
    run()
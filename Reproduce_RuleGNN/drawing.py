from scripts.WeightVisualization import WeightVisualization


def main():
    db_name = 'EXAMPLE_DB'
    main_config_file = 'Examples/CustomExample/Configs/config_main.yml'
    experiment_config_file = 'Examples/CustomExample/Configs/config_experiment.yml'
    graph_ids = [0, 1, 2]
    ww = WeightVisualization(db_name=db_name, main_config=main_config_file, experiment_config=experiment_config_file)
    ww.visualize(graph_ids, run=0, validation_id=0, draw_type='kawai')

if __name__ == '__main__':
    main()
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from find_best_models import config_paths_to_absolute, load_preprocessed_data_and_parameters
from src.Architectures.RuleGNN import RuleGNN
from src.utils.GraphData import get_graph_data
from src.utils.Parameters.Parameters import Parameters
from src.utils.RunConfiguration import get_run_configs
from src.utils.load_splits import Load_Splits


class RunSavedModel:
    def __init__(self, db_name, main_config, experiment_config, data_format='NEL'):
        self.db_name = db_name
        try:
            main_config_datasets = yaml.load(open(main_config), Loader=yaml.FullLoader)
            for dataset in main_config_datasets['datasets']:
                if dataset['name'] == db_name:
                    self.main_config = dataset
                    break
        except FileNotFoundError:
            print(f"Main config file {main_config} not found")
            return
        try:
            self.experiment_config = yaml.load(open(experiment_config), Loader=yaml.FullLoader)
        except FileNotFoundError:
            print(f"Experiment config file {experiment_config} not found")
            return
        self.data_format = data_format
        # get the absolute path
        absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        absolute_path = Path(absolute_path)
        self.validation_folds = self.main_config['validation_folds']
        # get the data path from the config file
        config_paths_to_absolute(self.experiment_config, absolute_path)
        self.results_path = self.experiment_config['paths']['results'].joinpath(db_name).joinpath('Results')
        self.m_path = self.experiment_config['paths']['results'].joinpath(db_name).joinpath('Models')
        self.graph_data = get_graph_data(db_name=db_name,
                                         data_path=self.experiment_config['paths']['data'],
                                         input_features=self.experiment_config.get('input_features', None),
                                         graph_format=data_format)


    def run(self, run=0, validation_id=0):
        # adapt the precision of the input data
        if self.experiment_config.get('precision', 'double') == 'double':
            for i in range(len(self.graph_data.input_data)):
                self.graph_data.input_data[i] = self.graph_data.input_data[i].double()


        run_configs = get_run_configs(self.experiment_config)

        for i, run_config in enumerate(run_configs):
            config_id = str(i).zfill(6)
            model_path = self.m_path.joinpath(f'model_Best_Configuration_{config_id}_run_{run}_val_step_{validation_id}.pt')
            # check if the model exists
            if model_path.exists():
                with open(model_path, 'r'):
                    seed = validation_id + self.validation_folds * run
                    split_data = Load_Splits(self.experiment_config['paths']['splits'], self.db_name)
                    test_data = np.asarray(split_data[0][validation_id], dtype=int)
                    para = Parameters()
                    load_preprocessed_data_and_parameters(config_id=config_id, run_id=run, validation_id=validation_id,
                                                          graph_db_name=self.db_name, graph_data=self.graph_data,
                                                          run_config=run_config, para=para, validation_folds=self.validation_folds)

                    """
                        Get the first index in the results directory that is not used
                    """
                    para.set_file_index(size=6)
                    net = RuleGNN.RuleGNN(graph_data=self.graph_data,
                                          para=para,
                                          seed=seed, device=run_config.config.get('device', 'cpu'))

                    net.load_state_dict(torch.load(model_path, weights_only=True))

                    # iterate over the layers and print information
                    for i, layer in enumerate(net.net_layers):
                        layer.print_all()

                    # evaluate the performance of the model on the test data
                    outputs = torch.zeros((len(test_data), self.graph_data.num_classes), dtype=torch.double)
                    with torch.no_grad():
                        for j, data_pos in enumerate(test_data, 0):
                            inputs = torch.DoubleTensor(self.graph_data.input_data[data_pos])
                            outputs[j] = net(inputs, data_pos)
                        labels = self.graph_data.output_data[test_data]
                        # calculate the errors between the outputs and the labels by getting the argmax of the outputs and the labels
                        counter = 0
                        correct = 0
                        for i, x in enumerate(outputs, 0):
                            if torch.argmax(x) == torch.argmax(labels[i]):
                                correct += 1
                            counter += 1
                        accuracy = correct / counter
                        print(f"Accuracy for model {model_path} is {accuracy}")


def main():
    db_name = 'PTC_FM'
    main_config_file = 'Testing/Pruning/Configs/config_main.yml'
    experiment_config_file = 'Testing/Pruning/Configs/config_experiment.yml'
    run_saved_model = RunSavedModel(db_name=db_name, main_config=main_config_file, experiment_config=experiment_config_file)
    run_saved_model.run(0,0)

if __name__ == '__main__':
    main()
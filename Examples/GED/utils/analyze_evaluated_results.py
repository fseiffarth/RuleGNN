import os

import torch
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class EditOperation():
    def __init__(self, operation_string):
        self.operation_string = operation_string
        self.node = None
        self.edge = None
        self.operation_type = None
        self.operation_element = None
        parts = operation_string.split(' ')
        if len(parts) == 3:
            self.operation_element = parts[0]
            self.operation_value = parts[1]
            self.operation_type = parts[2]
            if self.operation_element == 'NODE':
                self.node = int(self.operation_value)
            elif self.operation_element == 'EDGE':
                u_v = self.operation_value.split('--')
                if len(u_v) == 2:
                    self.edge = (int(u_v[0]), int(u_v[1]))

def LoadPathResultsFromTxt(file_path):
    # output should be a list of dicts with keys 'source_id', 'step_id', 'target_id', 'operation', 'target_value'
    data = dict()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split('\t')
        data = []
        for line in lines[1:]:
            parts = line.strip().split('\t')
            source_id = int(parts[0])
            step_id = parts[1]
            if step_id != 'S':
                step_id = int(step_id)
            else:
                step_id = -1
            target_id = int(parts[2])
            output_value = parts[3].split(' ')
            # convert output_value to torch tensor
            output_value = torch.tensor([float(v) for v in output_value])
            operation_string = parts[4]
            operation = None
            if operation_string != 'NONE':
                operation = EditOperation(operation_string)
            entry = {'source_id': source_id, 'step_id': step_id, 'target_id': target_id, 'operation': operation, 'output_value': output_value}
            data.append(entry)
    return data

def LoadPathResultsFromPt(file_path):
    # output should be a list of dicts with keys 'source_id', 'step_id', 'target_id', 'operation', 'target_value'
    data = torch.load(file_path, weights_only=False)
    return data

def LoadResultFromTxt(file_path):
    # output should be a dict where the keys are the graph ids and the values a dict with keys 'target_value' and 'output_value' and their corresponding values
    data = dict()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split('\t')
        data = []
        for line in lines[1:]:
            parts = line.strip().split('\t')
            # parts are graph_id, target_value and rest are output values
            graph_id = int(parts[0])
            target_value = float(parts[1])
            output_value = parts[2].split(' ')
            # convert output_value to torch tensor
            output_value = torch.tensor([float(v) for v in output_value])
            max_index = torch.argmax(output_value).item()
            entry = {'graph_id': graph_id, 'target_label': target_value, 'output_value': output_value, 'predicted_label': max_index, 'is_correct': (max_index == int(target_value))}
            data.append(entry)
    return data

class TrainingResults():
    def __init__(self, file_path):
        pass

class MainEvaluation():
    # initialize
    def __init__(self, path, strategy, dataset_name):

        # check if path/main_evaluation_dataset_name_strategy exists
        self.path = path
        self.strategy = strategy
        self.dataset_name = dataset_name
        main_evaluation_file = os.path.join(path, f'main_evaluation_{dataset_name}_{strategy}_meta.pt')
        if not os.path.exists(main_evaluation_file):
            # get all files in the path that match the strategy and dataset_name
            files = os.listdir(path)
            strategy_files = [f for f in files if strategy in f and dataset_name in f]
            dataset_files = [f for f in files if dataset_name in f]
            # separate files into path_results_files, training_results_files, validation_results_files
            self.path_results_files = [f for f in strategy_files if 'path_target_values' in f and f.endswith('.pt')]
            self.path_results_files_pt = [f for f in strategy_files if 'target_output' in f and f.endswith('.pt')]
            self.training_results_files = [f for f in dataset_files if 'train_results' in f]
            self.validation_results_files = [f for f in dataset_files if 'validation_results' in f]


            self.training_results = {}
            self.validation_results = {}
            self.path_results = {}
            self.path_results_pt = {}

            for file in self.path_results_files_pt:
                parts = file.split('_')
                config_part = [p for p in parts if p.startswith('config')]
                val_part = [p for p in parts if p.startswith('val')]
                if config_part and val_part:
                    config_id = int(config_part[0].replace('config', ''))
                    val_id = int(val_part[0].replace('val', ''))
                    file_path = os.path.join(path, file)
                    self.path_results_pt[(config_id, val_id)] = LoadPathResultsFromPt(file_path)

            # Load the training results in a dict with key as (config_id, val_id)
            for file in self.training_results_files:
                parts = file.split('_')
                config_part = [p for p in parts if p.startswith('config')]
                val_part = [p for p in parts if p.startswith('val')]
                if config_part and val_part:
                    config_id = int(config_part[0].replace('config', ''))
                    val_id = int(val_part[0].replace('val', ''))
                    file_path = os.path.join(path, file)
                    self.training_results[(config_id, val_id)] = LoadResultFromTxt(file_path)
            # Load the validation results in a dict with key as (config_id, val_id)
            for file in self.validation_results_files:
                parts = file.split('_')
                config_part = [p for p in parts if p.startswith('config')]
                val_part = [p for p in parts if p.startswith('val') and 'validation' not in p]
                if config_part and val_part:
                    config_id = int(config_part[0].replace('config', ''))
                    val_id = int(val_part[0].replace('val', ''))
                    file_path = os.path.join(path, file)
                    self.validation_results[(config_id, val_id)] = LoadResultFromTxt(file_path)
            # Load the path results in a dict with key as (config_id, val_id)
            for file in self.path_results_files:
                parts = file.split('_')
                config_part = [p for p in parts if p.startswith('config')]
                val_part = [p for p in parts if p.startswith('val') and 'values' not in p]
                if config_part and val_part:
                    config_id = int(config_part[0].replace('config', ''))
                    val_id = int(val_part[0].replace('val', ''))
                    file_path = os.path.join(path, file)
                    self.path_results[(config_id, val_id)] = LoadPathResultsFromTxt(file_path)
            # save the loaded results as a binary file
            # save the loaded results as binary files (optimized split-save)
            try:
                self.save()
            except Exception:
                # if save fails, continue without raising to preserve previous behavior
                pass
        else:
            # load the saved evaluation object
            # prefer split files for faster parallel load if available, otherwise support legacy single-file
            loaded = False
            try:
                # try split files
                loaded = self._load_split_files(path, dataset_name, strategy)
            except Exception:
                loaded = False

            if not loaded:
                # If split files weren't usable, try to rebuild from raw text files (fast text -> structured)
                rebuilt = False
                try:
                    rebuilt = self._build_from_txt(path)
                except Exception:
                    rebuilt = False

                if not rebuilt:
                    # fallback to legacy single-file
                    saved_file = os.path.join(path, f'main_evaluation_{dataset_name}_{strategy}.pt')
                    saved_data = torch.load(saved_file, weights_only=False)
                    self.training_results = saved_data['training_results']
                    self.validation_results = saved_data['validation_results']
                    self.path_results = saved_data['path_results']

    def _split_base(self):
        return os.path.join(self.path, f'main_evaluation_{self.dataset_name}_{self.strategy}')

    def _save_file(self, obj, filepath):
        # Use pickle highest protocol for fastest serialization and ensure torch handles tensors
        torch.save(obj, filepath, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def _load_file(self, filepath):
        # map to cpu to avoid GPU tensors issues and speed
        return torch.load(filepath, map_location='cpu')

    def save(self):
        """Save results in parallel into three files (training/validation/path) plus a small meta file.
        This is faster than a single large pickle in many environments and allows parallel IO.
        """
        base = self._split_base()
        training_file = base + '_training.pt'
        validation_file = base + '_validation.pt'
        path_file = base + '_path.pt'
        meta_file = base + '_meta.pt'

        meta = {'created_at': time.time(), 'version': 1}

        # Write meta first (small) then dump the three large objects in parallel
        with open(meta_file, 'wb') as fh:
            pickle.dump(meta, fh, protocol=pickle.HIGHEST_PROTOCOL)

        tasks = []
        with ThreadPoolExecutor(max_workers=3) as ex:
            tasks.append(ex.submit(self._save_file, self.training_results, training_file))
            tasks.append(ex.submit(self._save_file, self.validation_results, validation_file))
            tasks.append(ex.submit(self._save_file, self.path_results, path_file))
            # wait for completion and propagate exceptions
            for fut in as_completed(tasks):
                fut.result()

    def _load_split_files(self, path, dataset_name, strategy) -> bool:
        base = os.path.join(path, f'main_evaluation_{dataset_name}_{strategy}')
        training_file = base + '_training.pt'
        validation_file = base + '_validation.pt'
        path_file = base + '_path.pt'
        meta_file = base + '_meta.pt'

        if not (os.path.exists(training_file) and os.path.exists(validation_file) and os.path.exists(path_file)):
            return False

        # if any of the split files is empty, consider split format invalid and fall back
        try:
            if os.path.getsize(training_file) == 0 or os.path.getsize(validation_file) == 0 or os.path.getsize(path_file) == 0:
                return False
        except OSError:
            return False

        # load files in parallel (IO-bound)
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {
                'training': ex.submit(self._load_file, training_file),
                'validation': ex.submit(self._load_file, validation_file),
                'path': ex.submit(self._load_file, path_file)
            }
            # collect results
            # propagate exceptions so the caller can fallback to legacy load
            try:
                self.training_results = futs['training'].result()
                self.validation_results = futs['validation'].result()
                self.path_results = futs['path'].result()
            except Exception:
                # make sure to cancel remaining futures
                for k, f in futs.items():
                    if not f.done():
                        f.cancel()
                raise

        # optional: read meta if needed
        try:
            if os.path.exists(meta_file):
                with open(meta_file, 'rb') as fh:
                    _ = pickle.load(fh)
        except Exception:
            pass

        return True

    def _build_from_txt(self, path) -> bool:
        """Parse available raw text results in `path` and populate training/validation/path dicts.
        Returns True on success, False if no suitable files were found.
        """
        try:
            files = os.listdir(path)
        except Exception:
            return False

        strategy_files = [f for f in files if self.strategy in f and self.dataset_name in f]
        dataset_files = [f for f in files if self.dataset_name in f]

        training_results_files = [f for f in dataset_files if 'train_results' in f]
        validation_results_files = [f for f in dataset_files if 'validation_results' in f]
        path_results_files = [f for f in strategy_files if 'path_target_values' in f]

        if not (training_results_files or validation_results_files or path_results_files):
            return False

        training_results = {}
        validation_results = {}
        path_results = {}

        for file in training_results_files:
            parts = file.split('_')
            config_part = [p for p in parts if p.startswith('config')]
            val_part = [p for p in parts if p.startswith('val')]
            if config_part and val_part:
                config_id = int(config_part[0].replace('config', ''))
                val_id = int(val_part[0].replace('val', ''))
                file_path = os.path.join(path, file)
                training_results[(config_id, val_id)] = LoadResultFromTxt(file_path)

        for file in validation_results_files:
            parts = file.split('_')
            config_part = [p for p in parts if p.startswith('config')]
            val_part = [p for p in parts if p.startswith('val') and 'validation' not in p]
            if config_part and val_part:
                config_id = int(config_part[0].replace('config', ''))
                val_id = int(val_part[0].replace('val', ''))
                file_path = os.path.join(path, file)
                validation_results[(config_id, val_id)] = LoadResultFromTxt(file_path)

        for file in path_results_files:
            parts = file.split('_')
            config_part = [p for p in parts if p.startswith('config')]
            val_part = [p for p in parts if p.startswith('val') and 'values' not in p]
            if config_part and val_part:
                config_id = int(config_part[0].replace('config', ''))
                val_id = int(val_part[0].replace('val', ''))
                file_path = os.path.join(path, file)
                path_results[(config_id, val_id)] = LoadPathResultsFromTxt(file_path)

        # assign and attempt to save split binaries
        self.training_results = training_results
        self.validation_results = validation_results
        self.path_results = path_results
        try:
            self.save()
        except Exception:
            # ignore save errors but return success since parsing succeeded
            pass
        return True

    def analyzeFold(self, config_id, val_id):
        # analyze the results for the given config_id and val_id
        training_results = self.training_results.get((config_id, val_id), [])
        validation_results = self.validation_results.get((config_id, val_id), [])
        path_results = self.path_results.get((config_id, val_id), [])

        # print some statistics
        if training_results:
            correct = sum(1 for r in training_results if r['is_correct'])
            total = len(training_results)
            print(f"Training Results for config {config_id}, val {val_id}: {correct}/{total} correct, Accuracy: {correct/total:.4f}")
        else:
            print(f"No Training Results for config {config_id}, val {val_id}")

        if validation_results:
            correct = sum(1 for r in validation_results if r['is_correct'])
            total = len(validation_results)
            print(f"Validation Results for config {config_id}, val {val_id}: {correct}/{total} correct, Accuracy: {correct/total:.4f}")
        else:
            print(f"No Validation Results for config {config_id}, val {val_id}")

        if path_results:
            print(f"Path Results for config {config_id}, val {val_id}: {len(path_results)} entries loaded")
        else:
            print(f"No Path Results for config {config_id}, val {val_id}")

if __name__ == '__main__':
    config_id = 0
    val_id = 0
    db = 'MUTAG'
    strategy = 'i-E_d-IsoN'
    gnn_algorithm = 'GATv2'
    results_path = f'Examples/GED/Results/{gnn_algorithm}/Evaluation/target_output_config{config_id}_val{val_id}_{db}_{strategy}.pt'
    mainEvaluation = MainEvaluation(f'Examples/GED/Results/{gnn_algorithm}/Evaluation', strategy, db)
    mainEvaluation.analyzeFold(config_id, val_id)
    pass
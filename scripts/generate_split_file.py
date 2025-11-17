# This python script generates a split file for a dataset, dividing it into training, validation, and test sets.
# The split ratios can be customized as needed and also supports stratified splitting based on class labels.
# Moreover, custom splits can be used and exported to the specified output file.
# It can load the processed dataset to extract necessary information for splitting or the number of graphs in the dataset can be provided directly.
import argparse
import os
import random
import json
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from typing import Optional, Sequence, List, Dict, Any


def _try_load_labels_from_torch(path: str) -> Optional[np.ndarray]:
    """Try to load a PyTorch dataset/pt file and extract labels if possible.

    This is best-effort: it tries common attributes (.y, .labels) or iterates items
    if the object is indexable. Returns None if it cannot infer labels.
    """
    try:
        import torch
    except Exception:
        return None

    try:
        data = torch.load(path)
    except Exception:
        return None

    # If it's a dataset-like object with length and __getitem__
    try:
        # If it's a simple tensor/array with shape (N, ...), try to detect labels
        if hasattr(data, "y"):
            labels = data.y
            return np.array(labels).reshape(-1)
        if isinstance(data, (list, tuple)):
            labels = []
            for i, item in enumerate(data):
                # common patterns: item.y, item.label, item[1]
                if hasattr(item, "y"):
                    labels.append(item.y)
                elif hasattr(item, "label"):
                    labels.append(item.label)
                elif isinstance(item, (list, tuple)) and len(item) > 1:
                    labels.append(item[1])
                else:
                    # cannot extract labels for this element
                    labels = None
                    break
            if labels is not None:
                return np.array(labels).reshape(-1)
    except Exception:
        return None

    return None


def generate_splits(
    num_graphs: Optional[int] = None,
    labels: Optional[Sequence[int]] = None,
    n_folds: int = 10,
    val_size: Optional[float] = 0.1,
    val_count: Optional[int] = None,
    stratify: bool = False,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate a list of fold dictionaries matching the MUTAG_splits.json format.

    Arguments:
    - num_graphs: total number of graphs (required if `labels` not provided and no dataset load is done externally).
    - labels: optional sequence of length num_graphs with class labels (for stratified splitting).
    - n_folds: number of outer folds (default 10).
    - val_size: fraction of the training part to reserve for validation (ignored if val_count is set).
    - val_count: exact number of validation examples (overrides val_size when provided).
    - stratify: whether to perform stratified splits for both outer folds and inner validation (requires `labels`).
    - seed: random seed for reproducibility.

    Returns:
    - A list of length `n_folds` where each element is a dict with keys:
      {"test": [...], "model_selection": [ {"train": [...], "validation": val_list}] }

    The returned indices are plain Python lists (JSON-serializable) and are sorted ascending inside each list.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if labels is not None:
        labels = np.array(labels).reshape(-1)
        if num_graphs is None:
            num_graphs = len(labels)
    if num_graphs is None:
        raise ValueError("Either num_graphs or labels must be provided.")

    indices = np.arange(num_graphs)

    # Choose outer splitter
    if stratify and (labels is not None):
        outer_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        outer_iter = outer_splitter.split(indices, labels)
    else:
        outer_splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        outer_iter = outer_splitter.split(indices)

    folds = []
    for train_val_idx, test_idx in outer_iter:
        # train_val_idx: indices used for model selection (train+val)
        # test_idx: held-out test indices
        # Now split train_val into train and validation
        if val_count is not None:
            # Validate val_count against current fold size to give a clear error
            if val_count >= len(train_val_idx):
                raise ValueError(
                    f"val_count ({val_count}) must be smaller than the model-selection set size ({len(train_val_idx)}) for each fold"
                )
            # calculate val_size fraction equivalent for train_test_split convenience
            # but we'll use train_test_split with stratify if needed to select val_count
            if stratify and (labels is not None):
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=val_count,
                    stratify=labels[train_val_idx],
                    random_state=seed,
                )
            else:
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=val_count,
                    random_state=seed,
                )
        else:
            # val_size is a fraction
            if val_size is None or val_size <= 0:
                # no validation split requested
                train_idx = train_val_idx
                val_idx = np.array([], dtype=int)
            else:
                if stratify and (labels is not None):
                    train_idx, val_idx = train_test_split(
                        train_val_idx,
                        test_size=val_size,
                        stratify=labels[train_val_idx],
                        random_state=seed,
                    )
                else:
                    train_idx, val_idx = train_test_split(
                        train_val_idx,
                        test_size=val_size,
                        random_state=seed,
                    )

        # sort for deterministic output similar to the example file
        test_list = sorted(int(x) for x in np.unique(test_idx))
        train_list = sorted(int(x) for x in np.unique(train_idx))
        val_list = sorted(int(x) for x in np.unique(val_idx))

        fold = {"test": test_list, "model_selection": [{"train": train_list, "validation": val_list}]}
        folds.append(fold)

    return folds


def save_splits_json(splits: List[Dict[str, Any]], output_path: str) -> None:
    """Save the generated splits to JSON using the same layout as the example file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(splits, fh)


def main():
    parser = argparse.ArgumentParser(description="Generate dataset split JSON compatible with the project's format.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--num-graphs", type=int, help="Number of graphs in the dataset")
    group.add_argument("--dataset-path", type=str, help="Path to a dataset file (best-effort to extract labels and length)")
    parser.add_argument("--labels-path", type=str, default=None, help="Optional path to a JSON/NPY file with labels (overrides dataset label extraction)")
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--val-size", type=float, default=0.1, help="Fraction of model-selection set to use as validation (ignored if --val-count present)")
    parser.add_argument("--val-count", type=int, default=None, help="Exact number of validation examples (overrides --val-size)")
    parser.add_argument("--stratify", action="store_true", help="Perform stratified splits (requires labels)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: Data/Splits/<NAME>_splits.json)")
    parser.add_argument("--no-test", action="store_true", help="Do not create a test split; only create a train/validation split")

    args = parser.parse_args()

    labels = None
    num_graphs = args.num_graphs

    if args.dataset_path is not None:
        # try to infer labels and number of graphs
        labels = _try_load_labels_from_torch(args.dataset_path)
        if labels is not None:
            num_graphs = len(labels)
        else:
            # fallback: if file is a numpy array or json list of labels
            if args.labels_path is None:
                # try to load labels from dataset_path if it's a plain json/np file
                try:
                    if args.dataset_path.endswith('.json'):
                        with open(args.dataset_path, 'r') as f:
                            obj = json.load(f)
                            if isinstance(obj, list):
                                labels = np.array(obj)
                                num_graphs = len(labels)
                    elif args.dataset_path.endswith('.npy'):
                        labels = np.load(args.dataset_path)
                        num_graphs = len(labels)
                except Exception:
                    pass

    if args.labels_path is not None:
        # explicit labels file
        try:
            if args.labels_path.endswith('.json'):
                with open(args.labels_path, 'r') as f:
                    labels = np.array(json.load(f))
            elif args.labels_path.endswith('.npy'):
                labels = np.load(args.labels_path)
            else:
                # try plain text, one label per line
                with open(args.labels_path, 'r') as f:
                    labels = np.array([int(line.strip()) for line in f if line.strip()])
        except Exception as e:
            raise RuntimeError(f"Failed to load labels from {args.labels_path}: {e}")
        num_graphs = len(labels)

    if num_graphs is None:
        raise RuntimeError("Could not determine number of graphs. Provide --num-graphs or a loadable --dataset-path.")

    # If user requested no test split, produce a single train/validation split covering the whole dataset.
    if args.no_test:
        # Produce n_folds validation folds covering the whole dataset.
        if args.stratify and (labels is None):
            raise RuntimeError("Stratified splitting requested but no labels available. Provide --labels-path or a dataset with labels.")

        # If user provided val_size/val_count, they are irrelevant in no-test mode using n_folds
        if args.val_count is not None or (args.val_size is not None and args.val_size > 0):
            print("Note: --no-test set: ignoring --val-size and --val-count; using --n-folds to produce validation folds")

        if args.n_folds < 1:
            raise ValueError("--n-folds must be >= 1")
        if args.n_folds > num_graphs:
            raise ValueError(f"--n-folds ({args.n_folds}) cannot be greater than the dataset size ({num_graphs})")

        indices = np.arange(num_graphs)
        folds = []
        if args.stratify and (labels is not None):
            splitter = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
            split_iter = splitter.split(indices, labels)
        else:
            splitter = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
            split_iter = splitter.split(indices)

        for train_idx, val_idx in split_iter:
            train_list = sorted(int(x) for x in np.unique(train_idx))
            val_list = sorted(int(x) for x in np.unique(val_idx))
            folds.append({"test": [], "model_selection": [{"train": train_list, "validation": val_list}]})

        splits = folds
    else:
        splits = generate_splits(
            num_graphs=num_graphs,
            labels=labels,
            n_folds=args.n_folds,
            val_size=args.val_size,
            val_count=args.val_count,
            stratify=args.stratify,
            seed=args.seed,
        )

    if args.output is None:
        # derive a sensible default name
        base = 'custom'
        if args.dataset_path:
            base = os.path.splitext(os.path.basename(args.dataset_path))[0]
        elif args.num_graphs:
            base = f"n{args.num_graphs}"
        out = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data', 'Splits', f"{base}_splits.json")
    else:
        out = args.output

    save_splits_json(splits, out)
    print(f"Saved splits to {out}")


if __name__ == '__main__':
    main()

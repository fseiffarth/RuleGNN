# GIN splits

These JSON split files contain the train/validation splits used in the experiments reported in the paper
"HOW POWERFUL ARE GRAPH NEURAL NETWORKS?" (Xu et al., ICLR 2019). They mirror the format used throughout
`Data/Splits/*_splits.json` in this repository.

Format
- The file is a JSON array of outer folds (typically 10). Each element is an object with keys:
  - `test`: list of integer graph indices used for the outer test fold.
  - `model_selection`: list with a single object containing `train` and `validation` lists of indices used for
    inner model selection (training and validation).

Notes
- Indices are zero-based and refer to the ordering of graphs in the dataset used in the referenced paper.
- If you need to regenerate similar splits, see `scripts/generate_splits.py` which produces the same structure.


This repository is the implementation of HIAL. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Training

To train the model(s) in the paper:
```python
python HIAL.py
```

### Arguments
- `--use_dataset`: Specify the dataset to use. Options: `Cora`, `Citeseer`, `PubMed`, `DBLP`, etc. Default: `Citeseer`.
- `--radium`: Radius for coreset selection. Default: `0.005`.
- `--alpha`, `--beta`, `--gamma`: Hyperparameters for HIAL framwork. Default: `0.5`.
- `--prop_layer`: Number of propagation layers. Default: `1`.
- `--K`: Label budget of active learning. Default: `20`.
- `--use_propagation`: Propagation method. Options: `HGNN`, `UniGCN`, etc. Default: `HOIK`.
- `--dropout`: Dropout rate. Default: `0.6`.
- `--hidden_size`: Hidden layer size. Default: `64`.
- `--lr`: Learning rate. Default: `0.01`.
- `--weight_decay`: Weight decay for optimizer. Default: `5e-4`.

For more details, see `hyperparameters.py` or use `python HIAL.py --help`.


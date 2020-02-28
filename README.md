# IFT-6759-Team-13
Project 1: Advanced Projects in Machine Learning

Preprocessing:
```
cd prepro
python data_cleaning.py
cd ..
python -m utilities.dataloader2 --data_catalog_path='prepro/clean_df'
```
## Training:

1. Run command:

```
python utilities/data_cleaning_no_melt.py
```

This will create train_df, valid_df, and test_df

2. Run command: 

`python utilities/dataloader_simple2.py`

It takes ~4 hours to create x,y cache. This enables very fast training and validation.

3. Run command:

`python simple_keras_cnn.py`

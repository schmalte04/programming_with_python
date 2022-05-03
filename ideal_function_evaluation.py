import pandas as pd

train_input = pd.read_csv('data/train.csv', index_col=0)
test_input = pd.read_csv('data/test.csv', index_col=0)
ideal_input = pd.read_csv('data/ideal.csv', index_col=0)


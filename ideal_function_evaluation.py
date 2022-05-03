import pandas as pd
import sqlalchemy as db

train_input = pd.read_csv('data/train.csv', index_col=0)
test_input = pd.read_csv('data/test.csv', index_col=0)
ideal_input = pd.read_csv('data/ideal.csv', index_col=0)

# Defining the Engine
engine = db.create_engine('sqlite:///users.db', echo=True)
  
# Create the Metadata Object
metadata_obj = db.MetaData()

train_input.to_sql('train_data', engine)
ideal_input.to_sql('ideal_fucntion',engine)
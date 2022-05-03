import pandas as pd
import sqlalchemy as db
import numpy as np


#### Frist step: Read the CSV files and store them as objects
train_input = pd.read_csv('data/train.csv', index_col=0)
test_input = pd.read_csv('data/test.csv', index_col=0)
ideal_input = pd.read_csv('data/ideal.csv', index_col=0)

# Next step: Upload the tables to SQL database
#engine = db.create_engine('sqlite:///users.db', echo=True)
  
#metadata_obj = db.MetaData()
#train_input.to_sql('train_data', engine)

x=train_input[:,0]
y=train_input[:,1]


print(train_input.iloc[:,0:1])


glyph = Scatter(x="x", y="y", size="sizes", marker="square")
plot.add_glyph(source, glyph)
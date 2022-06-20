import pandas as pd
import sqlalchemy as db
import numpy as np
import matplotlib.pyplot as plt




#### Frist step: Read the CSV files and store them as objects
train_input = pd.read_csv('data/train.csv')
test_input = pd.read_csv('data/test.csv')
ideal_input = pd.read_csv('data/ideal.csv')

# Next step: Upload the tables to SQL database
#engine = db.create_engine('sqlite:///users.db', echo=True)
  
#metadata_obj = db.MetaData()
#train_input.to_sql('train_data', engine)

df_results_best_function= pd.DataFrame({'y1': ["NaN"],'y2': ["NaN"],'y3': ["NaN"],'y4': ["NaN"]}, index=['Best_Function'])
print(df_results_best_function)

df_least_deviation = pd.DataFrame(index=ideal_input.columns, columns=["Sum_Squared_Deviation"])

train_set = df_results_best_function.columns

print(train_set)

for e in train_set: 

        l = ideal_input.columns
        for i in l[1:]:
            df_least_deviation.Sum_Squared_Deviation[i] = sum((train_input[e]-ideal_input[i])**2)    

        df_least_deviation['Sum_Squared_Deviation'] = pd.to_numeric(df_least_deviation['Sum_Squared_Deviation'])
        df_results_best_function[e] = df_least_deviation['Sum_Squared_Deviation'].idxmin()

print(df_results_best_function)
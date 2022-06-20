import pandas as pd
import sqlalchemy as db
import numpy as np
import matplotlib.pyplot as plt
import math
import unittest


#### Frist step: Read the CSV files and store them as objects
train_input = pd.read_csv('data/train.csv')
test_input = pd.read_csv('data/test.csv')
ideal_input = pd.read_csv('data/ideal.csv')

# Next step: Upload the tables to SQL database
#engine = db.create_engine('sqlite:///users.db', echo=True)
  
#metadata_obj = db.MetaData()
#train_input.to_sql('train_data', engine)

df_results_best_function= pd.DataFrame({'y1': ["NaN"],'y2': ["NaN"],'y3': ["NaN"],'y4': ["NaN"]}, index=['Best_Function'])

df_least_deviation = pd.DataFrame(index=ideal_input.columns, columns=["Sum_Squared_Deviation"])

train_set = df_results_best_function.columns

for e in train_set: 

        l = ideal_input.columns
        for i in l[1:]:
            df_least_deviation.Sum_Squared_Deviation[i] = sum((train_input[e]-ideal_input[i])**2)    

        df_least_deviation['Sum_Squared_Deviation'] = pd.to_numeric(df_least_deviation['Sum_Squared_Deviation'])
        df_results_best_function[e] = df_least_deviation['Sum_Squared_Deviation'].idxmin()




#### Now check if the chosen functions fullfill the test requirements (max dev shall not exceed the max dev of test with ideal function by factor sqrt(2))

columns_best_functions = df_results_best_function.columns

for index in range(df_results_best_function.shape[1]):
    
    print(index)
    print(df_results_best_function.iloc[0,index])

    ideal_function= ideal_input.loc[:, ['x', df_results_best_function.iloc[0,index]]] 
    ### Map the 3 regressions by x using the merge function
    
    print(train_input)

    mapping_y1 = test_input.merge(ideal_function, on='x', how='left')
    mapping_y1 = mapping_y1.merge(train_input.iloc[:,[0,index+1]],on='x', how='left')

    #mapping_y1 = mapping_y1.assign(deviation_y21=lambda x: (x.y1-x), deviation_test=lambda x: (x['y']))

    mapping_y1['dev_ideal_function'] = mapping_y1.iloc[:,3] - mapping_y1.iloc[:,2]
    mapping_y1['dev_test_function'] = mapping_y1.iloc[:,3] - mapping_y1.iloc[:,1]

    max_ideal_function = mapping_y1['dev_ideal_function'].max()
    max_test_function = mapping_y1['dev_test_function'].max()

    print(max_ideal_function)
    print(max_test_function)

    if  max_ideal_function/max_test_function < math.sqrt(2):
        print("Result can be assigend and fullfills the criteria")
    
    else:
        print("Result does not fulfill the criteria")

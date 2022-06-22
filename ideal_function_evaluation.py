import pandas as pd
import sqlalchemy as db
import numpy as np
import matplotlib.pyplot as plt
import math
import unittest
from functions import ClassifierFunction


#### Frist step: Read the CSV files and store them as objects
train_input = pd.read_csv('data/train.csv')
test_input = pd.read_csv('data/test.csv')
ideal_input = pd.read_csv('data/ideal.csv')

# Next step: Upload the tables to SQL database
#engine = db.create_engine('sqlite:///users.db', echo=True)
  
#metadata_obj = db.MetaData()
#train_input.to_sql('train_data', engine)

df_results_best_function= pd.DataFrame({'y1': ["NaN","NaN","NaN"],'y2': ["NaN","NaN","NaN"],'y3': ["NaN","NaN","NaN"],'y4': ["NaN","NaN","NaN"]}, index=['Best_Function','MaxDeviation',"ClassificationThreshold"])

df_least_deviation = pd.DataFrame(index=ideal_input.columns[1:], columns=["Sum_Squared_Deviation","Deviation"])

train_set = df_results_best_function.columns
ideal_function_columns = ['x']

for e in train_set: 

        l = ideal_input.columns
        
        for i in l[1:]:

            df_least_deviation.Sum_Squared_Deviation[i] = sum((np.subtract(train_input[e],ideal_input[i])) **2)    
            df_least_deviation.Deviation[i] = (np.subtract(train_input[e],ideal_input[i]))   
        
        df_results_best_function.loc['Best_Function',e] = df_least_deviation['Sum_Squared_Deviation'].astype('float64').idxmin()
       
        ideal_function_columns.append(df_results_best_function.loc['Best_Function',e])

        df_deviations = pd.DataFrame({'train': train_input[e], 'ideal_function': ideal_input[df_least_deviation['Sum_Squared_Deviation'].astype('float64').idxmin()], 'deviation': np.subtract(train_input[e],ideal_input[df_least_deviation['Sum_Squared_Deviation'].astype('float64').idxmin()])})
       
        df_results_best_function.loc['MaxDeviation',e] = df_deviations['deviation'].max()
        df_results_best_function.loc['ClassificationThreshold',e] = df_deviations['deviation'].max()*math.sqrt(2)

           
SetIdealFunctionData = ideal_input.loc[:,ideal_function_columns]

test_input = test_input.merge(SetIdealFunctionData, on='x', how='left')

test_input['Classifier'] = "NaN"
test_input['Deviation'] = "NaN" 


### loop over ideal functions and check if classifier of test set is within threshold


Classifier_Results = ClassifierFunction(test_input,df_results_best_function)
print(Classifier_Results.to_markdown())
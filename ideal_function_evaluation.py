import pandas as pd
from sqlalchemy import create_engine,  column
import numpy as np
import matplotlib.pyplot as plt
import math
import unittest
from functions import ClassifierFunction
from functions import FindIdealFunction
from functions import CreateIdealFunctionHeader
from functions import utility
from functions import toSql
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import row, column

if __name__ == '__main__':

#### Frist step: Read the CSV files and store them as objects
    path_train = 'data/train.csv'
    path_test = 'data/test.csv'
    path_ideal = 'data/ideal.csv'

    ideal_input = utility(csvPath=path_ideal).csv
    train_input = utility(csvPath=path_train).csv
    test_input = utility(csvPath=path_test).csv

    # convert aideal and training files to sql using sqlalchemy
    toSql(obj=ideal_input,fileName="ideal", suffix=" (ideal function)")
    toSql(obj=train_input,fileName="training", suffix=" (training function)")


### Prepare Result dataframes to be passed to the function 
df_results_best_function= pd.DataFrame({'y1': ["NaN","NaN","NaN"],'y2': ["NaN","NaN","NaN"],'y3': ["NaN","NaN","NaN"],'y4': ["NaN","NaN","NaN"]}, index=['Best_Function','MaxDeviation',"ClassificationThreshold"])
df_least_deviation = pd.DataFrame(index=ideal_input.columns[1:], columns=["Sum_Squared_Deviation","Deviation"])

### Calculate Ideal Functions
df_results_best_function = FindIdealFunction(train_input,ideal_input,df_results_best_function,df_least_deviation)

output_file("training.html")
        
# instantiating the figure object
graph1 = figure(title = "Scatter Plot Training Data",x_axis_label='x', y_axis_label='y')
graph1.scatter(train_input.loc[:,'x'], train_input.loc[:,'y1'], color='#000000', size=10, legend_label='Training Data Y1')
graph1.scatter(ideal_input.loc[:,'x'], ideal_input.loc[:,'y21'], color='#0000ff', size=10, legend_label='Ideal Function')

graph2 = figure(title = "Scatter Plot Training Data",x_axis_label='x', y_axis_label='y')
graph2.scatter(train_input.loc[:,'x'], train_input.loc[:,'y2'], color='#000000', size=10, legend_label='Training Data Y2')
graph2.scatter(ideal_input.loc[:,'x'], ideal_input.loc[:,'y10'], color='#0000ff', size=10, legend_label='Ideal Function')

graph3 = figure(title = "Scatter Plot Training Data",x_axis_label='x', y_axis_label='y')
graph3.scatter(train_input.loc[:,'x'], train_input.loc[:,'y3'], color='#000000', size=10, legend_label='Training Data Y3')
graph3.scatter(ideal_input.loc[:,'x'], ideal_input.loc[:,'y18'], color='#0000ff', size=10, legend_label='Ideal Function')

graph4 = figure(title = "Scatter Plot Training Data",x_axis_label='x', y_axis_label='y')
graph4.scatter(train_input.loc[:,'x'], train_input.loc[:,'y4'], color='#000000', size=10, legend_label='Training Data Y4')
graph4.scatter(ideal_input.loc[:,'x'], ideal_input.loc[:,'y15'], color='#0000ff', size=10, legend_label='Ideal Function')

show(column(graph1, graph2,graph3,graph4))

### Create a Object with x and the names of the four ideal functions to extract them in the next step
ideal_function_columns = CreateIdealFunctionHeader(df_results_best_function)


SetIdealFunctionData = ideal_input.loc[:,ideal_function_columns]
test_input = test_input.merge(SetIdealFunctionData, on='x', how='left')


### Classifier Function:
'''
1) Calulates Deviation between Y_Test and Y_Ideal_Functions at point x
2) Checks weather deviation is smaller than the largest deviation of the ideal_functions with the training set by factor sqrt(2) "Critical Threshold
3) If a deviation between Y_Test and Y_Ideal_Function is within the threshold, it will be mapped

Result is a table with mapped Ideal Functions if possible for each Test Point given and the deviations.

Required Inputs: Test_Input and df_results_best_function (result from the first part of the assignment)
'''
Classifier_Results = ClassifierFunction(test_input,df_results_best_function)
print(Classifier_Results.to_markdown())


toSql(obj=Classifier_Results.loc[:,['x','y','Delta Y (test func)','No. of ideal func']],fileName="ClassifiedTest", suffix=" (test classification)")


#### last step: vizualize the data properly and save it



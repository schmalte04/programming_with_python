import pandas as pd
from sqlalchemy import create_engine, column
import numpy as np
import matplotlib.pyplot as plt
import math
import unittest
from functions import (
    ClassifierFunction,
    FindIdealFunction,
    CreateIdealFunctionHeader,
    toSql,
    utility,
    DataDescription,
    compareEvaluation
)
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import row, column
from plots import graphTrainIdeal, CreateScatterPlotsTrain, CreateMappingPlots

if __name__ == "__main__":

    #### Frist step: Read the CSV files and store them as objects
    path_train = "data/train.csv"
    path_test = "data/test.csv"
    path_ideal = "data/ideal.csv"

    ideal_input = utility(csvPath=path_ideal).csv
    train_input = utility(csvPath=path_train).csv
    test_input = utility(csvPath=path_test).csv

    # convert aideal and training files to sql using sqlalchemy
    toSql(obj=ideal_input, fileName="ideal", suffix=" (ideal function)")
    toSql(obj=train_input, fileName="training", suffix=" (training function)")

### Provide descriptive statistics for each of the training sets
print(DataDescription(train_input.iloc[:, 1]))
print(DataDescription(train_input.iloc[:, 2]))
print(DataDescription(train_input.iloc[:, 3]))
print(DataDescription(train_input.iloc[:, 4]))

### Prepare Result dataframes to be passed to the function
df_results_best_function = pd.DataFrame(
    {
        "y1": ["NaN", "NaN", "NaN", "NaN", "NaN"],
        "y2": ["NaN", "NaN", "NaN", "NaN", "NaN"],
        "y3": ["NaN", "NaN", "NaN", "NaN", "NaN"],
        "y4": ["NaN", "NaN", "NaN", "NaN", "NaN"],
    },
    index=[
        "Best_Function",
        "Least Squares",
        "MSE",
        "MaxDeviation",
        "ClassificationThreshold",
    ],
)

df_least_deviation = pd.DataFrame(
    index=ideal_input.columns[1:],
    columns=["Sum_Squared_Deviation", "Mean_Squared_Deviation", "Deviation"],
)

### Calculate Ideal Functions based on chosen evaluation method
# Ideal Function based on MSE
df_results_best_function_MSE = FindIdealFunction(
    train_input,
    ideal_input,
    df_results_best_function,
    df_least_deviation,
    evaluation="MSE",
)

print(df_results_best_function)

# Ideal Function based on  Least Square Method
df_results_best_function_LS = FindIdealFunction(
    train_input,
    ideal_input,
    df_results_best_function,
    df_least_deviation,
    evaluation="Least Squares",
)

compareEvaluation(df_results_best_function,df_results_best_function_LS)

#### As requested in the assignment we will work with the results of the Least Squares evaluation method
df_results_best_function = df_results_best_function_LS

#### create graphs with a for loop iterating through the training sets
graph_lst=[]

for f in df_results_best_function:
    print(f)
    graph = CreateScatterPlotsTrain(train_input, f)
    graph_lst.append(graph)

show(column(graph_lst[0],graph_lst[1],graph_lst[2],graph_lst[3]))

#### create graphs with a for loop iterating through the selected ideal functions sets

graph_best_function_lst=[]

for f in df_results_best_function:
    graph = graphTrainIdeal(
        train_input,
        ideal_input,
        f,
        df_results_best_function.loc["Best_Function", f],
        df_results_best_function.loc["Sum_Squared_Deviation", f],
        df_results_best_function.loc["MSE", f],
    )
    graph_best_function_lst.append(graph)
show(column(graph_best_function_lst[0],graph_best_function_lst[1],graph_best_function_lst[2],graph_best_function_lst[3]))



### Create a Object with x and the names of the four ideal functions to extract them in the next step
ideal_function_columns = CreateIdealFunctionHeader(df_results_best_function)
SetIdealFunctionData = ideal_input.loc[:, ideal_function_columns]
test_input = test_input.merge(SetIdealFunctionData, on="x", how="left")

### Classifier Function:
"""
1) Calulates Deviation between Y_Test and Y_Ideal_Functions at point x
2) Checks weather deviation is smaller than the largest deviation of the ideal_functions 
with the training set by factor sqrt(2) "Critical Threshold
3) If a deviation between Y_Test and Y_Ideal_Function is within the threshold, it will be mapped

Result is a table with mapped Ideal Functions if possible for each Test Point given and the deviations.

Required Inputs: Test_Input and df_results_best_function (result from the first part of the assignment)
"""

Classifier_Results = ClassifierFunction(test_input, df_results_best_function)


#print(Classifier_Results.to_markdown()) # print out the results of the classifier

## saves the classifier results as csv for the written assignment
save_results = Classifier_Results.to_csv("results/classifcation_results.csv")

toSql(
    obj=Classifier_Results.loc[
        :, ["x", "y", "Delta Y (test func)", "No. of ideal func"]
    ],
    fileName="ClassifiedTest",
    suffix=" (test classification)",
)


## Plot Ideal function results - functions are stored in plots.py
MappingPlot_y21 = CreateMappingPlots("y21", Classifier_Results, ideal_input)
MappingPlot_y10 = CreateMappingPlots("y10", Classifier_Results, ideal_input)
MappingPlot_y18 = CreateMappingPlots("y18", Classifier_Results, ideal_input)
MappingPlot_y15 = CreateMappingPlots("y15", Classifier_Results, ideal_input)

show(column(MappingPlot_y21, MappingPlot_y10, MappingPlot_y18, MappingPlot_y15))

import pandas as pd
import numpy as np
import math
import sqlalchemy as db
from sqlalchemy import create_engine

def ClassifierFunction(ToClassify, IdealFunction):

    ToClassify['No. of ideal func'] = ""
    ToClassify['Delta Y (test func)'] = "" 

    for i in range(2,5):
    
        for x_point in range(100):

            test_point_deviation = abs(np.subtract(ToClassify.y[x_point],ToClassify.iloc[x_point,i]))
            
            

            if test_point_deviation.max()<IdealFunction.iloc[2,i-2]:        
                ToClassify.iloc[x_point,6] = IdealFunction.iloc[0,i-2]
                ToClassify.iloc[x_point,7] = test_point_deviation.max().round(2)
            
            ToClassify.round(4)
            
    return(ToClassify)


def FindIdealFunction(train_input,ideal_input,df_results_best_function,df_least_deviation):

    train_set = df_results_best_function.columns

    for e in train_set: 

        l = ideal_input.columns
        
        for i in l[1:]:

            ### Subtract Training Y - Ideal Function Y and calc deviation and the sum of squared deviation
            df_least_deviation.Sum_Squared_Deviation[i] = sum((np.subtract(train_input[e],ideal_input[i])) **2)    
            df_least_deviation.Deviation[i] = (np.subtract(train_input[e],ideal_input[i]))   
        
        ### Check which Ideal Function has the lowest Sum of Squared Deviations
        df_results_best_function.loc['Best_Function',e] = df_least_deviation['Sum_Squared_Deviation'].astype('float64').idxmin()
       
        

        df_deviations = pd.DataFrame(
            {'train': train_input[e], 
            'ideal_function': ideal_input[df_least_deviation['Sum_Squared_Deviation'].astype('float64').idxmin()], 
            'deviation': np.subtract(train_input[e],ideal_input[df_least_deviation['Sum_Squared_Deviation'].astype('float64').idxmin()])})
       
        df_results_best_function.loc['MaxDeviation',e] = df_deviations['deviation'].max()
        df_results_best_function.loc['ClassificationThreshold',e] = df_deviations['deviation'].max()*math.sqrt(2)

    return(df_results_best_function)



def CreateIdealFunctionHeader(df):
    
    IdealFunctionList = ['x']
    
    for e in df:
        IdealFunctionList.append(df.loc['Best_Function',e])
    
    return(IdealFunctionList)



class utility:

    def __init__(self, csvPath):
        # csvpath param represents the input-data/filename
        # Here we are parsing the csv file into list of dataList and later iterate the object to get details
        # We need specific structure for csv in which 1st column is x and following column is y values
        self.dataFrames = []

        # The csv is being read by the Panda module and turned into a dataframe
        try:
            self.csv = pd.read_csv(csvPath)
        except FileNotFoundError:
            print("Could not read file {}".format(csvPath))
            raise


def toSql(obj, fileName, suffix):
        ###
        # This function accepts the filename and that is the name the db gets
        # the same function has suffix to specify based on original col name
        # Also, if the db file already exist in that case it will be replaced with new one
        # Here we are using sqlalchemy to handle all db related operations
        ###

        engineDB = db.create_engine('sqlite:///{}.db'.format(fileName), echo=False)

        # save file to db
        csvDataCopied = obj.copy()
        csvDataCopied.columns = [name.capitalize() + suffix for name in csvDataCopied.columns]
        csvDataCopied.set_index(csvDataCopied.columns[0], inplace=True)

        csvDataCopied.to_sql(
            fileName,
            engineDB,
            if_exists="replace",
            index=True,
        )



def createGraphFromTwoFunctions(scatterFunction, lineFunction):
    
    # first function dataframes and names
    functionOneDataframe = scatterFunction.dataframe
    functionOneName = scatterFunction.name

    # Second function dataframes and names
    functionTwoDataframe = lineFunction.dataframe
    functionTwoName = lineFunction.name

    graphPlot = figure(title="Graph for train model {} vs ideal {}.".format(functionOneName, functionTwoName),
               x_axis_label='x', y_axis_label='y')
    graphPlot.scatter(functionOneDataframe["x"], functionOneDataframe["y"], fill_color="green", legend_label="Train")
    graphPlot.line(functionTwoDataframe["x"], functionTwoDataframe["y"], legend_label="Ideal", line_width=5)
    return graphPlot
from statistics import mean, median
import pandas as pd
import numpy as np
import math
import sqlalchemy as db
from sqlalchemy import create_engine

def ClassifierFunction(ToClassify, IdealFunction):

    ToClassify['No. of ideal func'] = ""
    ToClassify['Delta Y (test func)'] = "" 

    for i in range(2,6):
    
        for x_point in range(100):
           
            test_point_deviation = abs(np.subtract(ToClassify.y[x_point],ToClassify.iloc[x_point,i]))

            if test_point_deviation<IdealFunction.iloc[4,i-2]:        
                
                ToClassify.iloc[x_point,6] = IdealFunction.iloc[0,i-2]
                ToClassify.iloc[x_point,7] = test_point_deviation.max().round(2)
            
            ToClassify.round(4)
            
    return(ToClassify)


def FindIdealFunction(train_input,ideal_input,df_results_best_function,df_least_deviation,evaluation):

    train_set = df_results_best_function.columns

    for e in train_set: 

        l = ideal_input.columns
        
        for i in l[1:]:

            ### Subtract Training Y - Ideal Function Y and calc deviation and the sum of squared deviation
            df_least_deviation.Sum_Squared_Deviation[i] = sum((np.subtract(train_input[e],ideal_input[i])) **2)
            df_least_deviation.Mean_Squared_Deviation[i] = sum((np.subtract(train_input[e],ideal_input[i])) **2)/len(train_input.index)
            df_least_deviation.Deviation[i] = (np.subtract(train_input[e],ideal_input[i]))   
        
        ### Check which Ideal Function has the lowest Sum of Squared Deviations
        
        if evaluation == 'Least Squares':
            df_results_best_function.loc['Best_Function',e] = df_least_deviation['Sum_Squared_Deviation'].astype('float64').idxmin()
            df_results_best_function.loc['Sum_Squared_Deviation',e] = df_least_deviation['Sum_Squared_Deviation'].astype('float64').min().round(2)
        elif evaluation == 'MSE':
            df_results_best_function.loc['Best_Function',e] = df_least_deviation['Mean_Squared_Deviation'].astype('float64').idxmin()

        df_results_best_function.loc['Least Squares',e] = df_least_deviation['Sum_Squared_Deviation'].astype('float64').min().round(2)
        df_results_best_function.loc['MSE',e] = df_least_deviation['Mean_Squared_Deviation'].astype('float64').min().round(2)



        df_deviations = pd.DataFrame(
            {'train': train_input[e], 
            'ideal_function': ideal_input[df_least_deviation['Sum_Squared_Deviation'].astype('float64').idxmin()], 
            'sum_squared_eror': df_least_deviation['Sum_Squared_Deviation'].astype('float64').idxmin(), 
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
       
        ''' 
        function to read csv files based on a given file path csvPath
        1) Create empty dataframe
        2) Read CSV Files and save in the precreated dataframe
        '''

        self.dataFrames = []

        try:
            self.csv = pd.read_csv(csvPath)
        except FileNotFoundError:
            print("Could not read file {}".format(csvPath))
            raise


def toSql(obj, fileName, suffix):

    try:   
       # Create the database engine with SQLAlchemy based on the given fileName
        engineDB = db.create_engine('sqlite:///{}.db'.format(fileName), echo=False)
    except:
        print("Could not create SQL Engine, check inputs")

    try:
        # save file to db with a pre defined 
        csvDataCopied = obj.copy()
        csvDataCopied.columns = [name.capitalize() + suffix for name in csvDataCopied.columns]
        csvDataCopied.set_index(csvDataCopied.columns[0], inplace=True)

        '''
        Uploading the respective object to a sql database
        '''
        csvDataCopied.to_sql(
            fileName,
            engineDB,
            if_exists="replace",
            index=True,
        )
    except: 
        print("Could not upload csv files to database")



def DataDescription(obj):
     
    result = pd.DataFrame(data={'Maximum': [0], 'Minimum': [0], 'Mean': [0],  'Median': [0]})
    result.loc[:,'Maximum']=obj.max()
    result.loc[:,'Minimum']=obj.min()
    result.loc[:,'Mean']=obj.mean()
    result.loc[:,'Median']=obj.median()

    return result


def createGraphFromTwoFunctions(scatterFunction, lineFunction):
    
    # first function dataframes and names
    functionOneDataframe = scatterFunction.dataframe
    functionOneName = scatterFunction.name

    # Second function dataframes and names
    functionTwoDataframe = lineFunction.dataframe
    functionTwoName = lineFunction.name

  
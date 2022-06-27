from bokeh.plotting import figure, output_file, show
from bokeh.layouts import row, column
from bokeh.models import Title


def graphTrainIdeal(trainData,idealData,TrainFunctName,IdealFunctionName, squaredError_LS,squaredError_MSE):
    graph2 = figure(x_axis_label='x', y_axis_label='y',plot_width=600,plot_height=300)
    graph2.scatter(trainData.loc[:,'x'], trainData.loc[:,TrainFunctName], color='#000000', size=4, legend_label='Training Data {}'.format(TrainFunctName))
    graph2.scatter(idealData.loc[:,'x'], idealData.loc[:,IdealFunctionName], color='blue', size=4, legend_label='Ideal Function Data {}'.format(IdealFunctionName), marker="x")
    graph2.add_layout(Title(text="Calculated Squared error = {}, Calculated MSE = {}".format(squaredError_LS,squaredError_MSE), text_font_style="italic"), 'above')
    graph2.add_layout(Title(text="Graph for train model {} vs ideal {}".format(TrainFunctName, IdealFunctionName), text_font_size="16pt"), 'above')
    return(graph2)    


def CreateScatterPlotsTrain(trainData,DataSetNumber):
    graph1 = figure(title = "Scatter Plot Training Data",x_axis_label='x', y_axis_label='y')
    graph1.scatter(trainData.loc[:,'x'], trainData.loc[:,DataSetNumber], color='#000000', size=10, legend_label='Training Data {}'.format(DataSetNumber))
    graph1.legend.location = "top_left"
    graph1.legend.background_fill_color = "white"
    graph1.legend.background_fill_alpha = 0.0
    graph1.add_layout(graph1.legend[0], 'above')

    return(graph1)

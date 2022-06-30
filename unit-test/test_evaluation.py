import unittest
from functions import FindIdealFunction
import pandas as pd


class UnitTestFindIdealFunction(unittest.TestCase):
    def test_MSE(self):
        """
        test if MSE is correct
        """

        train_input = pd.DataFrame({"x": [1, 2, 3], "y1": [1, 2, 3]})
        ideal_input = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 4], "yx": [10, 0, 3]})

        df_results_best_function = pd.DataFrame(
            {"y1": ["NaN", "NaN", "NaN", "NaN", "NaN"]},
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

        df_results_best_function = FindIdealFunction(
            train_input,
            ideal_input,
            df_results_best_function,
            df_least_deviation,
            evaluation="MSE",
        )
        result = df_results_best_function.loc["MSE", "y1"]
        self.assertEqual(result, 0.33, "The MSE should be  0.33")

    def test_IdealFunction(self):
        """
        test if it finds the correct ideal function
        """

        train_input = pd.DataFrame({"x": [1, 2, 3], "y1": [1, 2, 3]})
        ideal_input = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 4], "yx": [10, 0, 3]})

        df_results_best_function = pd.DataFrame(
            {"y1": ["NaN", "NaN", "NaN", "NaN", "NaN"]},
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

        df_results_best_function = FindIdealFunction(
            train_input,
            ideal_input,
            df_results_best_function,
            df_least_deviation,
            evaluation="MSE",
        )
        print(df_results_best_function)

        result = df_results_best_function.loc["Best_Function", "y1"]
        self.assertEqual(result, "y", "y should be the ideal function")

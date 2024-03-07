import os.path as path
import pandas as pd


def assert_msg(condition, msg):
    if not condition:
        raise Exception(msg)


def read_file(filename):
    # Get the absolute file path
    filepath = path.join(path.dirname(__file__), filename)

    # Check if the file exists
    assert_msg(path.exists(filepath), "File does not exist")

    # Read the CSV file and return
    return pd.read_csv(filepath,
                       index_col=0,
                       parse_dates=True,
                       infer_datetime_format=True)


def SMA(values, n):
    """
    Returns simple moving average
    """
    return pd.Series(values).rolling(n).mean()


def crossover(series1, series2) -> bool:
    """
    Check if two series cross over at the end
    :param series1:  Series 1
    :param series2:  Series 2
    :return:         Returns True if they cross, otherwise False
    """
    return series1[-2] < series2[-2] and series1[-1] > series2[-1]

"""
custom_metrics.py

This module contains custom metric functions for use in machine learning models.
These metrics are designed to work with LightGBM or other frameworks that support
custom evaluation metrics.

Author: Johan Mendez
Date: 2025-03-20
"""

import numpy as np

def smape(actual, predicted, datatype='darts'):
    """
    Calculates the sMAPE (Symmetric Mean Absolute Percentage Error) between actual and predicted.
    If 'datatype'=='darts', actual y predicted son TimeSeries de la librería Darts (y extraemos .values()).
    """
    if datatype == 'darts':
        actual = actual.values()
        predicted = predicted.values()

    numerator = np.abs(predicted - actual)
    denominator = np.abs(actual) + np.abs(predicted)


    with np.errstate(divide='ignore', invalid='ignore'):
        percentage_diff = numerator / denominator


    mask = (actual == 0) & (predicted == 0)
    percentage_diff[actual == 0] = np.nan

    smape_value = np.nanmean(percentage_diff)

    if np.isnan(smape_value):
        smape_value = np.nan

    return smape_value


def mape(actual, predicted, datatype='darts'):
    """
    Calculates the MAPE (Mean Absolute Percentage Error) between actual and predicted.
    If 'datatype'=='darts', actual y predicted son TimeSeries de la librería Darts (y extraemos .values()).
    """

    if datatype == 'darts':
        actual = actual.values()
        predicted = predicted.values()

    numerator = actual - predicted
    denominator = actual

    with np.errstate(divide='ignore', invalid='ignore'):
        percentage_diff = np.abs(numerator / denominator)


    mask = (actual == 0) & (predicted != 0)
    percentage_diff[mask] = np.nan

    mape_value = np.nanmean(percentage_diff)

    if np.isnan(mape_value):
        mape_value = np.nan

    return mape_value

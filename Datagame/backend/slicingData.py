import app
import pandas as pd
import numpy as np

def slice (data):
    cols=data.columns

    num_cols=data._get_numeric_data().columns
    cat_cols=list(set(cols)-set(num_cols))

    return num_cols, cat_cols


    
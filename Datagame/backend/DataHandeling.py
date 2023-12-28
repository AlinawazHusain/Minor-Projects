import app
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from backend import slicingData



def fillna(df):
    num,cat=slicingData.slice(df)
    imputer1=SimpleImputer(missing_values=np.nan,strategy="mean")
    imputer2=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
    ct=ColumnTransformer([('simpleimputer',imputer1,num),('simpleimputer2',imputer2,cat)])
    df=ct.fit_transform(df)
    return df

    

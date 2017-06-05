import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

df = pd.read_csv('diamonds.csv')
df.head()
df.describe()
df.cut.unique()
df.color.unique()
df.clarity.unique()

#check for Null Values
df.isnull().values.any()
#no need to impute, but impute by group
#df.fillna(df.mean()) #imputes ALL columns with mean for each respectivlely
#df["ColumnX"].fillna(df["ColumnX"].mean(), inplace=True) #mean for specific column
#df["ColumnX"].fillna(df.groupby("ColumnZ")["ColumnX"].transform("mean"), inplace=True) #imputes by group
df.columns

#split into matricies for sklearn
col=[u'carat', u'cut', u'color', u'clarity', u'depth', u'tbl', u'price',u'x', u'y', u'z']
X = df.loc[:, col].values
y = df.loc[:, 'price'].values
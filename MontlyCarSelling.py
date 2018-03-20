#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 02:01:50 2018

@author: furkancoskun
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA


df_2004 = pd.read_excel("MODELDOKUMUARALIK2004.xls",skiprows=1,index_col=None)
df_2005 = pd.read_excel("MODELDOKUMUARALIK2005.xls",skiprows=1,index_col=None)
df_2006 = pd.read_excel("Model Dökümü, Aralık 2006.xls",skiprows=1,index_col=None)
df_2007 = pd.read_excel("MODELDOKUMARALIK2007.xls",skiprows=1,index_col=None)
df_2008 = pd.read_excel("MODELDOKUMUARALIK2008.xls",skiprows=1,index_col=None)
df_2009 = pd.read_excel("MODELDOKUMUARALIK2009.xls",skiprows=1,index_col=None)
df_2010 = pd.read_excel("MODELDOKUMUARALIK2010.xls",skiprows=1,index_col=None)
df_2011 = pd.read_excel("MODELDOKUMUARALIK2011.xls",skiprows=1,index_col=None)
df_2012 = pd.read_excel("MODELDOKUMARALIK2012.xls",skiprows=1,index_col=None)
df_2013 = pd.read_excel("MODELDOKUMUARALIK2013.xls",skiprows=1,index_col=None)
df_2014 = pd.read_excel("MODELDOKUMUARALIK2014.xls",skiprows=1,index_col=None)
df_2015 = pd.read_excel("MODELDOKUMARALIK2015.xls",skiprows=2,index_col=None)
df_2016 = pd.read_excel("Model Dokumu Aralık'2016.xls",skiprows=2,index_col=None)
df_2017 = pd.read_excel("Model Dokumu Aralık 2017.xlsx",skiprows=2,index_col=None)
df_2018 = pd.read_excel("Model Dokumu Şubat'2018.xls",skiprows=2,index_col=None)

df_2004.info()
df_2004.values
df_2004.values.dtype


def segment(df):
    df = df.iloc[:-25,3:16]
    df['SEGMENT'] = df['SEGMENT'].astype('str')
    df = df[df.SEGMENT.str.contains("nan") == False]
    df['SEGMENT'] = df['SEGMENT'].apply(lambda x: str(x)[0])
    df = df.groupby(['SEGMENT']).sum()
    return df


df_2004 = segment(df_2004)
df_2005 = segment(df_2005)
df_2006 = segment(df_2006)
df_2007 = segment(df_2007)
df_2008 = segment(df_2008)
df_2009 = segment(df_2009)
df_2010 = segment(df_2010)
df_2011 = segment(df_2011)
df_2012 = segment(df_2012)
df_2013 = segment(df_2013)
df_2014 = segment(df_2014)
df_2015 = segment(df_2015)
df_2016 = segment(df_2016)
df_2017 = segment(df_2017)
df_2018 = segment(df_2018)

df_2018 = df_2018.iloc[:,0:2]


df_2004 = df_2004.transpose()
df_2005 = df_2005.transpose()
df_2006 = df_2006.transpose()
df_2007 = df_2007.transpose()
df_2008 = df_2008.transpose()
df_2009 = df_2009.transpose()
df_2010 = df_2010.transpose()
df_2011 = df_2011.transpose()
df_2012 = df_2012.transpose()
df_2013 = df_2013.transpose()
df_2014 = df_2014.transpose()
df_2015 = df_2015.transpose()
df_2016 = df_2016.transpose()
df_2017 = df_2017.transpose()
df_2018 = df_2018.transpose()




data = pd.concat([df_2004,df_2005,df_2006,df_2007,df_2008,df_2009,df_2010,df_2011,df_2012,df_2013,df_2014,df_2015,df_2016,df_2017,df_2018])
del df_2004,df_2005,df_2006,df_2007,df_2008,df_2009,df_2010,df_2011,df_2012,df_2013,df_2014,df_2015,df_2016,df_2017,df_2018



autocorrelation_plot(data)






#Shifting to get last month and last2month
data2 = data.copy()
data2 = data2.shift(1)

data['Last_Month_A'] = data2['A']
data['Last_Month_B'] = data2['B']
data['Last_Month_C'] = data2['C']
data['Last_Month_D'] = data2['D']
data['Last_Month_E'] = data2['E']
data['Last_Month_F'] = data2['F']

del data2



usd = pd.read_excel("usd_try_monthly.xls",skiprows = 1, index_col = None)
usd = usd.set_index('Tarih')
eur = pd.read_excel("eur_try_monthly.xls",skiprows = 1, index_col = None)
eur = eur.set_index('Tarih')
bist = pd.read_excel("bist100monthly.xls",skiprows = 1, index_col = None)
bist = bist.set_index('Tarih')
export_import = pd.read_excel("Export_Import.xls")
export_import = export_import.set_index('Month')


#data = data.join(usd)
#data = data.join(eur)
#data = data.join(bist)
#data = data.join(export_import)



'''
    
    SHIFTED FEUTURES

'''

usd = usd.join(eur)
del eur
usd = usd.join(bist)
del bist
usd = usd.join(export_import)
del export_import



last2usd = usd.copy()


usd['USD Diff -1'] = usd['USD Diff'].fillna(usd['USD Diff'])
usd['EUR Diff -1'] = usd['EUR Diff'].fillna(usd['EUR Diff'])
usd['BIST Diff -1'] = usd['BIST Diff'].fillna(usd['BIST Diff'])



#for -3 month
last2usd = last2usd.shift(1)
usd['Export Diff -1'] = 100*(usd['Exports'] - last2usd['Exports'])/usd['Exports']
usd['Import Diff -1'] = 100*(usd['Imports'] - last2usd['Imports'])/usd['Imports']



#for -2 month
last2usd = last2usd.shift(1)
usd['USD Diff -2'] = (100*(usd['USD Now'] - last2usd['USD Now'])/usd['USD Now']).fillna(usd['USD Diff -1'])
usd['EUR Diff -2'] = (100*(usd['EUR Now'] - last2usd['EUR Now'])/usd['EUR Now']).fillna(usd['EUR Diff -1'])
usd['BIST Diff -2'] = (100*(usd['BIST Now'] - last2usd['BIST Now'])/usd['BIST Now']).fillna(usd['BIST Diff -1'])
usd['Export Diff -2'] = (100*(usd['Exports'] - last2usd['Exports'])/usd['Exports']).fillna(usd['Export Diff -1'])
usd['Import Diff -2'] = (100*(usd['Imports'] - last2usd['Imports'])/usd['Imports']).fillna(usd['Import Diff -1'])



#for -3 month
last2usd = last2usd.shift(1)
usd['USD Diff -3'] = (100*(usd['USD Now'] - last2usd['USD Now'])/usd['USD Now']).fillna(usd['USD Diff -2'])
usd['EUR Diff -3'] = (100*(usd['EUR Now'] - last2usd['EUR Now'])/usd['EUR Now']).fillna(usd['EUR Diff -2'])
usd['BIST Diff -3'] = (100*(usd['BIST Now'] - last2usd['BIST Now'])/usd['BIST Now']).fillna(usd['BIST Diff -2'])
usd['Export Diff -3'] = (100*(usd['Exports'] - last2usd['Exports'])/usd['Exports']).fillna(usd['Export Diff -2'])
usd['Import Diff -3'] = (100*(usd['Imports'] - last2usd['Imports'])/usd['Imports']).fillna(usd['Import Diff -2'])

#for -6 month
last2usd = last2usd.shift(3)
usd['USD Diff -6'] = (100*(usd['USD Now'] - last2usd['USD Now'])/usd['USD Now']).fillna(usd['USD Diff -3'])
usd['EUR Diff -6'] = (100*(usd['EUR Now'] - last2usd['EUR Now'])/usd['EUR Now']).fillna(usd['EUR Diff -3'])
usd['BIST Diff -6'] = (100*(usd['BIST Now'] - last2usd['BIST Now'])/usd['BIST Now']).fillna(usd['BIST Diff -3'])
usd['Export Diff -6'] = (100*(usd['Exports'] - last2usd['Exports'])/usd['Exports']).fillna(usd['Export Diff -3'])
usd['Import Diff -6'] = (100*(usd['Imports'] - last2usd['Imports'])/usd['Imports']).fillna(usd['Import Diff -3'])

#for -9 month
last2usd = last2usd.shift(3)
usd['USD Diff -9'] = (100*(usd['USD Now'] - last2usd['USD Now'])/usd['USD Now']).fillna(usd['USD Diff -6'])
usd['EUR Diff -9'] = (100*(usd['EUR Now'] - last2usd['EUR Now'])/usd['EUR Now']).fillna(usd['EUR Diff -6'])
usd['BIST Diff -9'] = (100*(usd['BIST Now'] - last2usd['BIST Now'])/usd['BIST Now']).fillna(usd['BIST Diff -6'])
usd['Export Diff -9'] = (100*(usd['Exports'] - last2usd['Exports'])/usd['Exports']).fillna(usd['Export Diff -6'])
usd['Import Diff -9'] = (100*(usd['Imports'] - last2usd['Imports'])/usd['Imports']).fillna(usd['Import Diff -6'])

#for -12 month
last2usd = last2usd.shift(3)
usd['USD Diff -12'] = (100*(usd['USD Now'] - last2usd['USD Now'])/usd['USD Now']).fillna(usd['USD Diff -9'])
usd['EUR Diff -12'] = (100*(usd['EUR Now'] - last2usd['EUR Now'])/usd['EUR Now']).fillna(usd['EUR Diff -9'])
usd['BIST Diff -12'] = (100*(usd['BIST Now'] - last2usd['BIST Now'])/usd['BIST Now']).fillna(usd['BIST Diff -9'])
usd['Export Diff -12'] = (100*(usd['Exports'] - last2usd['Exports'])/usd['Exports']).fillna(usd['Export Diff -9'])
usd['Import Diff -12'] = (100*(usd['Imports'] - last2usd['Imports'])/usd['Imports']).fillna(usd['Import Diff -9'])


usd = usd.shift(-1)
usd = usd.iloc[:,18:]

data = data.iloc[:,0:12]
data = data.join(usd)
del usd, last2usd




'''
    
    SHIFTED FEATURES

'''







#sns.pairplot(data, x_vars=['BIST Now',
#                           'BIST Open',
#                           'BIST High',
#                           'BIST Low',
#                           'BIST Vol',
#                           'BIST Diff',
#                           'USD Now'
#                           'USD Open',
#                           'USD High',
#                           'USD Low',
#                           'USD Diff',
#                           'EUR Now',
#                           'EUR Open',
#                           'EUR High',
#                           'EUR Low',
#                           'EUR Diff',
#                           'A',
#                           'B',
#                           'D',
#                           'E',
#                           'F',], y_vars='C',size=7, aspect=0.7,kind='reg')




X_train = data.iloc[1:150,6:]
X_test = data.iloc[150:-1,6:]
    
y_trainset = data.iloc[1:150,:6]
y_testset = data.iloc[150:-1,:6]

y_A_train = y_trainset['A']
y_A_test = y_testset['A']
y_B_train = y_trainset['B']
y_B_test = y_testset['B']
y_C_train = y_trainset['C']
y_C_test = y_testset['C']
y_D_train = y_trainset['D']
y_D_test = y_testset['D']
y_E_train = y_trainset['E']
y_E_test = y_testset['E']
y_F_train = y_trainset['F']
y_F_test = y_testset['F']
del y_trainset, y_testset



model = ARIMA(data, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())























def rmsle(ytrue,ypred):
    return np.sqrt(mean_squared_log_error(ytrue,ypred))



from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor    
rf = RandomForestRegressor(n_estimators=100,oob_score=True)


feature_cols = X_train.columns.values
features = data[feature_cols]


rf.fit(X_train,y_A_train)
predicted_A = rf.predict(X_test)
print("%.4F" % rf.oob_score_)
p_A = y_A_test.values
error = rmsle(predicted_A,p_A)
del p_A
print "A, ",error
del error

importances = rf.feature_importances_
indices = np.argsort(importances)
feature_cols = feature_cols[indices]
feat_imp_A = pd.DataFrame(feature_cols)
feat_imp_A.sort_index(ascending=False,inplace=True)



rf.fit(X_train,y_B_train)
predicted_B = rf.predict(X_test)
print("%.4F" % rf.oob_score_)
p_B = y_B_test.values
error = rmsle(predicted_B,p_B)
del p_B
print "B, ",error
del error

importances = rf.feature_importances_
indices = np.argsort(importances)
feature_cols = feature_cols[indices]
feat_imp_B = pd.DataFrame(feature_cols)
feat_imp_B.sort_index(ascending=False,inplace=True)



rf.fit(X_train,y_C_train)
predicted_C = rf.predict(X_test)
print("%.4F" % rf.oob_score_)
p_C = y_C_test.values
error = rmsle(predicted_C,p_C)
del p_C
print "C, ",error
del error

importances = rf.feature_importances_
indices = np.argsort(importances)
feature_cols = feature_cols[indices]
feat_imp_C = pd.DataFrame(feature_cols)
feat_imp_C.sort_index(ascending=False,inplace=True)



rf.fit(X_train,y_D_train)
predicted_D = rf.predict(X_test)
print("%.4F" % rf.oob_score_)
p_D = y_D_test.values
error = rmsle(predicted_D,p_D)
del p_D
print "D, ",error
del error

importances = rf.feature_importances_
indices = np.argsort(importances)
feature_cols = feature_cols[indices]
feat_imp_D = pd.DataFrame(feature_cols)
feat_imp_D.sort_index(ascending=False,inplace=True)



rf.fit(X_train,y_E_train)
predicted_E = rf.predict(X_test)
print("%.4F" % rf.oob_score_)
p_E = y_E_test.values
error = rmsle(predicted_E,p_E)
del p_E
print "E, ",error
del error

importances = rf.feature_importances_
indices = np.argsort(importances)
feature_cols = feature_cols[indices]
feat_imp_E = pd.DataFrame(feature_cols)
feat_imp_E.sort_index(ascending=False,inplace=True)



rf.fit(X_train,y_F_train)
predicted_F = rf.predict(X_test)
print("%.4F" % rf.oob_score_)
p_F = y_F_test.values
error = rmsle(predicted_F,p_F)
del p_F
print "F, ",error
del error

importances = rf.feature_importances_
indices = np.argsort(importances)
feature_cols = feature_cols[indices]
feat_imp_F = pd.DataFrame(feature_cols)
feat_imp_F.sort_index(ascending=False,inplace=True)
del feature_cols, features, importances, indices





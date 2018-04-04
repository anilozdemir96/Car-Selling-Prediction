#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 02:01:50 2018

@author: furkancoskun
"""

import pandas as pd
import numpy as np

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



#autocorrelation_plot(data)






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

data['A Sell Diff'] = (100*(data['A'] - data['Last_Month_A'])/data['Last_Month_A'])
data['B Sell Diff'] = (100*(data['B'] - data['Last_Month_B'])/data['Last_Month_B'])
data['C Sell Diff'] = (100*(data['C'] - data['Last_Month_C'])/data['Last_Month_C'])
data['D Sell Diff'] = (100*(data['D'] - data['Last_Month_D'])/data['Last_Month_D'])
data['E Sell Diff'] = (100*(data['E'] - data['Last_Month_E'])/data['Last_Month_E'])
data['F Sell Diff'] = (100*(data['F'] - data['Last_Month_F'])/data['Last_Month_F'])


#for -1 month
last2usd = last2usd.shift(1)
data2 = data.copy().shift(1)
usd['Export Diff -1'] = 100*(usd['Exports'] - last2usd['Exports'])/last2usd['Exports']
usd['Import Diff -1'] = 100*(usd['Imports'] - last2usd['Imports'])/last2usd['Imports']

data['A Sell Diff -1'] = 100*(data['Last_Month_A'] - data2['Last_Month_A'])/data2['Last_Month_A']
data['B Sell Diff -1'] = 100*(data['Last_Month_B'] - data2['Last_Month_B'])/data2['Last_Month_B']
data['C Sell Diff -1'] = 100*(data['Last_Month_C'] - data2['Last_Month_C'])/data2['Last_Month_C']
data['D Sell Diff -1'] = 100*(data['Last_Month_D'] - data2['Last_Month_D'])/data2['Last_Month_D']
data['E Sell Diff -1'] = 100*(data['Last_Month_E'] - data2['Last_Month_E'])/data2['Last_Month_E']
data['F Sell Diff -1'] = 100*(data['Last_Month_F'] - data2['Last_Month_F'])/data2['Last_Month_F']





#for -2 month
last2usd = last2usd.shift(1)
data2 = data.copy().shift(1)

usd['USD Diff -2'] = (100*(usd['USD Now'] - last2usd['USD Now'])/last2usd['USD Now']).fillna(usd['USD Diff -1'])
usd['EUR Diff -2'] = (100*(usd['EUR Now'] - last2usd['EUR Now'])/last2usd['EUR Now']).fillna(usd['EUR Diff -1'])
usd['BIST Diff -2'] = (100*(usd['BIST Now'] - last2usd['BIST Now'])/last2usd['BIST Now']).fillna(usd['BIST Diff -1'])
usd['Export Diff -2'] = (100*(usd['Exports'] - last2usd['Exports'])/last2usd['Exports']).fillna(usd['Export Diff -1'])
usd['Import Diff -2'] = (100*(usd['Imports'] - last2usd['Imports'])/last2usd['Imports']).fillna(usd['Import Diff -1'])


data['A Sell Diff -2'] = (100*(data['Last_Month_A'] - data2['Last_Month_A'])/data2['Last_Month_A']).fillna(data['A Sell Diff -1'])
data['B Sell Diff -2'] = (100*(data['Last_Month_B'] - data2['Last_Month_B'])/data2['Last_Month_B']).fillna(data['B Sell Diff -1'])
data['C Sell Diff -2'] = (100*(data['Last_Month_C'] - data2['Last_Month_C'])/data2['Last_Month_C']).fillna(data['C Sell Diff -1'])
data['D Sell Diff -2'] = (100*(data['Last_Month_D'] - data2['Last_Month_D'])/data2['Last_Month_D']).fillna(data['D Sell Diff -1'])
data['E Sell Diff -2'] = (100*(data['Last_Month_E'] - data2['Last_Month_E'])/data2['Last_Month_E']).fillna(data['E Sell Diff -1'])
data['F Sell Diff -2'] = (100*(data['Last_Month_F'] - data2['Last_Month_F'])/data2['Last_Month_F']).fillna(data['F Sell Diff -1'])


#for -3 month
last2usd = last2usd.shift(1)
data2 = data.copy().shift(1)

usd['USD Diff -3'] = (100*(usd['USD Now'] - last2usd['USD Now'])/last2usd['USD Now']).fillna(usd['USD Diff -2'])
usd['EUR Diff -3'] = (100*(usd['EUR Now'] - last2usd['EUR Now'])/last2usd['EUR Now']).fillna(usd['EUR Diff -2'])
usd['BIST Diff -3'] = (100*(usd['BIST Now'] - last2usd['BIST Now'])/last2usd['BIST Now']).fillna(usd['BIST Diff -2'])
usd['Export Diff -3'] = (100*(usd['Exports'] - last2usd['Exports'])/last2usd['Exports']).fillna(usd['Export Diff -2'])
usd['Import Diff -3'] = (100*(usd['Imports'] - last2usd['Imports'])/last2usd['Imports']).fillna(usd['Import Diff -2'])


data['A Sell Diff -3'] = (100*(data['Last_Month_A'] - data2['Last_Month_A'])/data2['Last_Month_A']).fillna(data['A Sell Diff -2'])
data['B Sell Diff -3'] = (100*(data['Last_Month_B'] - data2['Last_Month_B'])/data2['Last_Month_B']).fillna(data['B Sell Diff -2'])
data['C Sell Diff -3'] = (100*(data['Last_Month_C'] - data2['Last_Month_C'])/data2['Last_Month_C']).fillna(data['C Sell Diff -2'])
data['D Sell Diff -3'] = (100*(data['Last_Month_D'] - data2['Last_Month_D'])/data2['Last_Month_D']).fillna(data['D Sell Diff -2'])
data['E Sell Diff -3'] = (100*(data['Last_Month_E'] - data2['Last_Month_E'])/data2['Last_Month_E']).fillna(data['E Sell Diff -2'])
data['F Sell Diff -3'] = (100*(data['Last_Month_F'] - data2['Last_Month_F'])/data2['Last_Month_F']).fillna(data['F Sell Diff -2'])

#for -6 month
last2usd = last2usd.shift(3)
data2 = data.copy().shift(3)

usd['USD Diff -6'] = (100*(usd['USD Now'] - last2usd['USD Now'])/last2usd['USD Now']).fillna(usd['USD Diff -3'])
usd['EUR Diff -6'] = (100*(usd['EUR Now'] - last2usd['EUR Now'])/last2usd['EUR Now']).fillna(usd['EUR Diff -3'])
usd['BIST Diff -6'] = (100*(usd['BIST Now'] - last2usd['BIST Now'])/last2usd['BIST Now']).fillna(usd['BIST Diff -3'])
usd['Export Diff -6'] = (100*(usd['Exports'] - last2usd['Exports'])/last2usd['Exports']).fillna(usd['Export Diff -3'])
usd['Import Diff -6'] = (100*(usd['Imports'] - last2usd['Imports'])/last2usd['Imports']).fillna(usd['Import Diff -3'])


data['A Sell Diff -6'] = (100*(data['Last_Month_A'] - data2['Last_Month_A'])/data2['Last_Month_A']).fillna(data['A Sell Diff -3'])
data['B Sell Diff -6'] = (100*(data['Last_Month_B'] - data2['Last_Month_B'])/data2['Last_Month_B']).fillna(data['B Sell Diff -3'])
data['C Sell Diff -6'] = (100*(data['Last_Month_C'] - data2['Last_Month_C'])/data2['Last_Month_C']).fillna(data['C Sell Diff -3'])
data['D Sell Diff -6'] = (100*(data['Last_Month_D'] - data2['Last_Month_D'])/data2['Last_Month_D']).fillna(data['D Sell Diff -3'])
data['E Sell Diff -6'] = (100*(data['Last_Month_E'] - data2['Last_Month_E'])/data2['Last_Month_E']).fillna(data['E Sell Diff -3'])
data['F Sell Diff -6'] = (100*(data['Last_Month_F'] - data2['Last_Month_F'])/data2['Last_Month_F']).fillna(data['F Sell Diff -3'])


#for -9 month
last2usd = last2usd.shift(3)
data2 = data.copy().shift(3)

usd['USD Diff -9'] = (100*(usd['USD Now'] - last2usd['USD Now'])/last2usd['USD Now']).fillna(usd['USD Diff -6'])
usd['EUR Diff -9'] = (100*(usd['EUR Now'] - last2usd['EUR Now'])/last2usd['EUR Now']).fillna(usd['EUR Diff -6'])
usd['BIST Diff -9'] = (100*(usd['BIST Now'] - last2usd['BIST Now'])/last2usd['BIST Now']).fillna(usd['BIST Diff -6'])
usd['Export Diff -9'] = (100*(usd['Exports'] - last2usd['Exports'])/last2usd['Exports']).fillna(usd['Export Diff -6'])
usd['Import Diff -9'] = (100*(usd['Imports'] - last2usd['Imports'])/last2usd['Imports']).fillna(usd['Import Diff -6'])


data['A Sell Diff -9'] = (100*(data['Last_Month_A'] - data2['Last_Month_A'])/data2['Last_Month_A']).fillna(data['A Sell Diff -6'])
data['B Sell Diff -9'] = (100*(data['Last_Month_B'] - data2['Last_Month_B'])/data2['Last_Month_B']).fillna(data['B Sell Diff -6'])
data['C Sell Diff -9'] = (100*(data['Last_Month_C'] - data2['Last_Month_C'])/data2['Last_Month_C']).fillna(data['C Sell Diff -6'])
data['D Sell Diff -9'] = (100*(data['Last_Month_D'] - data2['Last_Month_D'])/data2['Last_Month_D']).fillna(data['D Sell Diff -6'])
data['E Sell Diff -9'] = (100*(data['Last_Month_E'] - data2['Last_Month_E'])/data2['Last_Month_E']).fillna(data['E Sell Diff -6'])
data['F Sell Diff -9'] = (100*(data['Last_Month_F'] - data2['Last_Month_F'])/data2['Last_Month_F']).fillna(data['F Sell Diff -6'])



#for -12 month
last2usd = last2usd.shift(3)
data2 = data.copy().shift(3)

usd['USD Diff -12'] = (100*(usd['USD Now'] - last2usd['USD Now'])/last2usd['USD Now']).fillna(usd['USD Diff -9'])
usd['EUR Diff -12'] = (100*(usd['EUR Now'] - last2usd['EUR Now'])/last2usd['EUR Now']).fillna(usd['EUR Diff -9'])
usd['BIST Diff -12'] = (100*(usd['BIST Now'] - last2usd['BIST Now'])/last2usd['BIST Now']).fillna(usd['BIST Diff -9'])
usd['Export Diff -12'] = (100*(usd['Exports'] - last2usd['Exports'])/last2usd['Exports']).fillna(usd['Export Diff -9'])
usd['Import Diff -12'] = (100*(usd['Imports'] - last2usd['Imports'])/last2usd['Imports']).fillna(usd['Import Diff -9'])


data['A Sell Diff -12'] = (100*(data['Last_Month_A'] - data2['Last_Month_A'])/data2['Last_Month_A']).fillna(data['A Sell Diff -9'])
data['B Sell Diff -12'] = (100*(data['Last_Month_B'] - data2['Last_Month_B'])/data2['Last_Month_B']).fillna(data['B Sell Diff -9'])
data['C Sell Diff -12'] = (100*(data['Last_Month_C'] - data2['Last_Month_C'])/data2['Last_Month_C']).fillna(data['C Sell Diff -9'])
data['D Sell Diff -12'] = (100*(data['Last_Month_D'] - data2['Last_Month_D'])/data2['Last_Month_D']).fillna(data['D Sell Diff -9'])
data['E Sell Diff -12'] = (100*(data['Last_Month_E'] - data2['Last_Month_E'])/data2['Last_Month_E']).fillna(data['E Sell Diff -9'])
data['F Sell Diff -12'] = (100*(data['Last_Month_F'] - data2['Last_Month_F'])/data2['Last_Month_F']).fillna(data['F Sell Diff -9'])


usd = usd.shift(-1)
usd = usd.iloc[:,18:]

data = data.join(usd)
del usd, last2usd, data2




'''
    
    SHIFTED FEATURES

'''





X_train = data.iloc[2:150,6:12].join(data.iloc[2:150,18:])
X_test = data.iloc[150:-1,6:12].join(data.iloc[150:-1,18:])
    
y_trainset = data.iloc[2:150,:6]
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



#model = ARIMA(data, order=(5,1,0))
#model_fit = model.fit(disp=0)
#print(model_fit.summary())




print("SALE PREDICTION ACCORDING TO LAST MOUNTHS SALE WITH RANDOM FOREST REGRESSOR")


def rmsle(ytrue,ypred):
    return np.sqrt(mean_squared_error(ytrue,ypred))



from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor    
rf = RandomForestRegressor(n_estimators=100,oob_score=True)


feature_cols = X_train.columns.values
features = data[feature_cols]


rf.fit(X_train,y_A_train)
predicted_A = rf.predict(X_test)
print("oob score: %.4F" % rf.oob_score_)
p_A = y_A_test.values
error = rmsle(predicted_A,p_A)
del p_A
print "Error on A prediction: ",error
del error

importances = rf.feature_importances_
indices = np.argsort(importances)
feature_cols = feature_cols[indices]
feat_imp_A = pd.DataFrame(feature_cols)
feat_imp_A.sort_index(ascending=False,inplace=True)



rf.fit(X_train,y_B_train)
predicted_B = rf.predict(X_test)
print("oob score: %.4F" % rf.oob_score_)
p_B = y_B_test.values
error = rmsle(predicted_B,p_B)
del p_B
print "Error on B prediction: ",error
del error

importances = rf.feature_importances_
indices = np.argsort(importances)
feature_cols = feature_cols[indices]
feat_imp_B = pd.DataFrame(feature_cols)
feat_imp_B.sort_index(ascending=False,inplace=True)



rf.fit(X_train,y_C_train)
predicted_C = rf.predict(X_test)
print("oob score: %.4F" % rf.oob_score_)
p_C = y_C_test.values
error = rmsle(predicted_C,p_C)
del p_C
print "Error on C prediction: ",error
del error

importances = rf.feature_importances_
indices = np.argsort(importances)
feature_cols = feature_cols[indices]
feat_imp_C = pd.DataFrame(feature_cols)
feat_imp_C.sort_index(ascending=False,inplace=True)



rf.fit(X_train,y_D_train)
predicted_D = rf.predict(X_test)
print("oob score: %.4F" % rf.oob_score_)
p_D = y_D_test.values
error = rmsle(predicted_D,p_D)
del p_D
print "Error on D prediction: ",error
del error

importances = rf.feature_importances_
indices = np.argsort(importances)
feature_cols = feature_cols[indices]
feat_imp_D = pd.DataFrame(feature_cols)
feat_imp_D.sort_index(ascending=False,inplace=True)



rf.fit(X_train,y_E_train)
predicted_E = rf.predict(X_test)
print("oob score: %.4F" % rf.oob_score_)
p_E = y_E_test.values
error = rmsle(predicted_E,p_E)
del p_E
print "Error on E prediction: ",error
del error

importances = rf.feature_importances_
indices = np.argsort(importances)
feature_cols = feature_cols[indices]
feat_imp_E = pd.DataFrame(feature_cols)
feat_imp_E.sort_index(ascending=False,inplace=True)



rf.fit(X_train,y_F_train)
predicted_F = rf.predict(X_test)
print("oob score: %.4F" % rf.oob_score_)
p_F = y_F_test.values
error = rmsle(predicted_F,p_F)
del p_F
print "Error on F prediction: ",error
del error

importances = rf.feature_importances_
indices = np.argsort(importances)
feature_cols = feature_cols[indices]
feat_imp_F = pd.DataFrame(feature_cols)
feat_imp_F.sort_index(ascending=False,inplace=True)
del feature_cols, features, importances, indices







'''
        PREDICTION OF CHANGE ACCORDING TO CLASSIFICATION OF SALE CHANGE
'''



classified = data.iloc[2:-1,12:].copy()


print("PREDICTION OF CHANGE ACCORDING TO CLASSIFICATION OF SALE CHANGE")

# Target Classification Binary

classified['A Sell Diff'] = pd.qcut(classified['A Sell Diff'],2,labels=[0,1]).astype(int)
classified['B Sell Diff'] = pd.qcut(classified['B Sell Diff'],2,labels=[0,1]).astype(int)
classified['C Sell Diff'] = pd.qcut(classified['C Sell Diff'],2,labels=[0,1]).astype(int)
classified['D Sell Diff'] = pd.qcut(classified['D Sell Diff'],2,labels=[0,1]).astype(int)
classified['E Sell Diff'] = pd.qcut(classified['E Sell Diff'],2,labels=[0,1]).astype(int)
classified['F Sell Diff'] = pd.qcut(classified['F Sell Diff'],2,labels=[0,1]).astype(int)


# Future Classification

classified['A Sell Diff -1'] = pd.qcut(classified['A Sell Diff -1'],5,labels=[1,2,3,4,5]).astype(int)
classified['A Sell Diff -2'] = pd.qcut(classified['A Sell Diff -2'],5,labels=[1,2,3,4,5]).astype(int)
classified['A Sell Diff -3'] = pd.qcut(classified['A Sell Diff -3'],5,labels=[1,2,3,4,5]).astype(int)
classified['A Sell Diff -6'] = pd.qcut(classified['A Sell Diff -6'],5,labels=[1,2,3,4,5]).astype(int)
classified['A Sell Diff -9'] = pd.qcut(classified['A Sell Diff -9'],5,labels=[1,2,3,4,5]).astype(int)
classified['A Sell Diff -12'] = pd.qcut(classified['A Sell Diff -12'],5,labels=[1,2,3,4,5]).astype(int)

classified['B Sell Diff -1'] = pd.qcut(classified['B Sell Diff -1'],5,labels=[1,2,3,4,5]).astype(int)
classified['B Sell Diff -2'] = pd.qcut(classified['B Sell Diff -2'],5,labels=[1,2,3,4,5]).astype(int)
classified['B Sell Diff -3'] = pd.qcut(classified['B Sell Diff -3'],5,labels=[1,2,3,4,5]).astype(int)
classified['B Sell Diff -6'] = pd.qcut(classified['B Sell Diff -6'],5,labels=[1,2,3,4,5]).astype(int)
classified['B Sell Diff -9'] = pd.qcut(classified['B Sell Diff -9'],5,labels=[1,2,3,4,5]).astype(int)
classified['B Sell Diff -12'] = pd.qcut(classified['B Sell Diff -12'],5,labels=[1,2,3,4,5]).astype(int)

classified['C Sell Diff -1'] = pd.qcut(classified['C Sell Diff -1'],5,labels=[1,2,3,4,5]).astype(int)
classified['C Sell Diff -2'] = pd.qcut(classified['C Sell Diff -2'],5,labels=[1,2,3,4,5]).astype(int)
classified['C Sell Diff -3'] = pd.qcut(classified['C Sell Diff -3'],5,labels=[1,2,3,4,5]).astype(int)
classified['C Sell Diff -6'] = pd.qcut(classified['C Sell Diff -6'],5,labels=[1,2,3,4,5]).astype(int)
classified['C Sell Diff -9'] = pd.qcut(classified['C Sell Diff -9'],5,labels=[1,2,3,4,5]).astype(int)
classified['C Sell Diff -12'] = pd.qcut(classified['C Sell Diff -12'],5,labels=[1,2,3,4,5]).astype(int)

classified['D Sell Diff -1'] = pd.qcut(classified['D Sell Diff -1'],5,labels=[1,2,3,4,5]).astype(int)
classified['D Sell Diff -2'] = pd.qcut(classified['D Sell Diff -2'],5,labels=[1,2,3,4,5]).astype(int)
classified['D Sell Diff -3'] = pd.qcut(classified['D Sell Diff -3'],5,labels=[1,2,3,4,5]).astype(int)
classified['D Sell Diff -6'] = pd.qcut(classified['D Sell Diff -6'],5,labels=[1,2,3,4,5]).astype(int)
classified['D Sell Diff -9'] = pd.qcut(classified['D Sell Diff -9'],5,labels=[1,2,3,4,5]).astype(int)
classified['D Sell Diff -12'] = pd.qcut(classified['D Sell Diff -12'],5,labels=[1,2,3,4,5]).astype(int)

classified['E Sell Diff -1'] = pd.qcut(classified['E Sell Diff -1'],5,labels=[1,2,3,4,5]).astype(int)
classified['E Sell Diff -2'] = pd.qcut(classified['E Sell Diff -2'],5,labels=[1,2,3,4,5]).astype(int)
classified['E Sell Diff -3'] = pd.qcut(classified['E Sell Diff -3'],5,labels=[1,2,3,4,5]).astype(int)
classified['E Sell Diff -6'] = pd.qcut(classified['E Sell Diff -6'],5,labels=[1,2,3,4,5]).astype(int)
classified['E Sell Diff -9'] = pd.qcut(classified['E Sell Diff -9'],5,labels=[1,2,3,4,5]).astype(int)
classified['E Sell Diff -12'] = pd.qcut(classified['E Sell Diff -12'],5,labels=[1,2,3,4,5]).astype(int)

classified['F Sell Diff -1'] = pd.qcut(classified['F Sell Diff -1'],5,labels=[1,2,3,4,5]).astype(int)
classified['F Sell Diff -2'] = pd.qcut(classified['F Sell Diff -2'],5,labels=[1,2,3,4,5]).astype(int)
classified['F Sell Diff -3'] = pd.qcut(classified['F Sell Diff -3'],5,labels=[1,2,3,4,5]).astype(int)
classified['F Sell Diff -6'] = pd.qcut(classified['F Sell Diff -6'],5,labels=[1,2,3,4,5]).astype(int)
classified['F Sell Diff -9'] = pd.qcut(classified['F Sell Diff -9'],5,labels=[1,2,3,4,5]).astype(int)
classified['F Sell Diff -12'] = pd.qcut(classified['F Sell Diff -12'],5,labels=[1,2,3,4,5]).astype(int)

classified['USD Diff -1'] = pd.qcut(classified['USD Diff -1'],5,labels=[1,2,3,4,5]).astype(int)
classified['USD Diff -2'] = pd.qcut(classified['USD Diff -2'],5,labels=[1,2,3,4,5]).astype(int)
classified['USD Diff -3'] = pd.qcut(classified['USD Diff -3'],5,labels=[1,2,3,4,5]).astype(int)
classified['USD Diff -6'] = pd.qcut(classified['USD Diff -6'],5,labels=[1,2,3,4,5]).astype(int)
classified['USD Diff -9'] = pd.qcut(classified['USD Diff -9'],5,labels=[1,2,3,4,5]).astype(int)
classified['USD Diff -12'] = pd.qcut(classified['USD Diff -12'],5,labels=[1,2,3,4,5]).astype(int)

classified['EUR Diff -1'] = pd.qcut(classified['EUR Diff -1'],5,labels=[1,2,3,4,5]).astype(int)
classified['EUR Diff -2'] = pd.qcut(classified['EUR Diff -2'],5,labels=[1,2,3,4,5]).astype(int)
classified['EUR Diff -3'] = pd.qcut(classified['EUR Diff -3'],5,labels=[1,2,3,4,5]).astype(int)
classified['EUR Diff -6'] = pd.qcut(classified['EUR Diff -6'],5,labels=[1,2,3,4,5]).astype(int)
classified['EUR Diff -9'] = pd.qcut(classified['EUR Diff -9'],5,labels=[1,2,3,4,5]).astype(int)
classified['EUR Diff -12'] = pd.qcut(classified['EUR Diff -12'],5,labels=[1,2,3,4,5]).astype(int)

classified['BIST Diff -1'] = pd.qcut(classified['BIST Diff -1'],5,labels=[1,2,3,4,5]).astype(int)
classified['BIST Diff -2'] = pd.qcut(classified['BIST Diff -2'],5,labels=[1,2,3,4,5]).astype(int)
classified['BIST Diff -3'] = pd.qcut(classified['BIST Diff -3'],5,labels=[1,2,3,4,5]).astype(int)
classified['BIST Diff -6'] = pd.qcut(classified['BIST Diff -6'],5,labels=[1,2,3,4,5]).astype(int)
classified['BIST Diff -9'] = pd.qcut(classified['BIST Diff -9'],5,labels=[1,2,3,4,5]).astype(int)
classified['BIST Diff -12'] = pd.qcut(classified['BIST Diff -12'],5,labels=[1,2,3,4,5]).astype(int)

classified['Export Diff -1'] = pd.qcut(classified['Export Diff -1'],5,labels=[1,2,3,4,5]).astype(int)
classified['Export Diff -2'] = pd.qcut(classified['Export Diff -2'],5,labels=[1,2,3,4,5]).astype(int)
classified['Export Diff -3'] = pd.qcut(classified['Export Diff -3'],5,labels=[1,2,3,4,5]).astype(int)
classified['Export Diff -6'] = pd.qcut(classified['Export Diff -6'],5,labels=[1,2,3,4,5]).astype(int)
classified['Export Diff -9'] = pd.qcut(classified['Export Diff -9'],5,labels=[1,2,3,4,5]).astype(int)
classified['Export Diff -12'] = pd.qcut(classified['Export Diff -12'],5,labels=[1,2,3,4,5]).astype(int)

classified['Import Diff -1'] = pd.qcut(classified['Import Diff -1'],5,labels=[1,2,3,4,5]).astype(int)
classified['Import Diff -2'] = pd.qcut(classified['Import Diff -2'],5,labels=[1,2,3,4,5]).astype(int)
classified['Import Diff -3'] = pd.qcut(classified['Import Diff -3'],5,labels=[1,2,3,4,5]).astype(int)
classified['Import Diff -6'] = pd.qcut(classified['Import Diff -6'],5,labels=[1,2,3,4,5]).astype(int)
classified['Import Diff -9'] = pd.qcut(classified['Import Diff -9'],5,labels=[1,2,3,4,5]).astype(int)
classified['Import Diff -12'] = pd.qcut(classified['Import Diff -12'],5,labels=[1,2,3,4,5]).astype(int)




X_train = classified.iloc[:150,6:]
X_test = classified.iloc[150:,6:]

y_trainset = classified.iloc[:150,:6]
y_testset = classified.iloc[150:,:6]

y_A_train = y_trainset['A Sell Diff']
y_A_test = y_testset['A Sell Diff']
y_B_train = y_trainset['B Sell Diff']
y_B_test = y_testset['B Sell Diff']
y_C_train = y_trainset['C Sell Diff']
y_C_test = y_testset['C Sell Diff']
y_D_train = y_trainset['D Sell Diff']
y_D_test = y_testset['D Sell Diff']
y_E_train = y_trainset['E Sell Diff']
y_E_test = y_testset['E Sell Diff']
y_F_train = y_trainset['F Sell Diff']
y_F_test = y_testset['F Sell Diff']
del y_trainset, y_testset


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)

rf.fit(X_train, y_A_train)
print "Random Forest Classifier For A"
print("oob Score: "+"%.4f" % rf.oob_score_)

predicted_A = rf.predict(X_test)

print ("Original A values:  " + str(y_A_test.values))
print ("Predicted A values: " + str(predicted_A))
print ("Error: " + str(np.mean(y_A_test.values != predicted_A)))



feature_cols = X_train.columns.values
features = data[feature_cols]


importances = rf.feature_importances_
indices = np.argsort(importances)
feature_cols = feature_cols[indices]
feat_rfc_A = pd.DataFrame(feature_cols)
feat_rfc_A.sort_index(ascending=False,inplace=True)






rf.fit(X_train, y_B_train)
print "Random Forest Classifier For B"
print("oob Score: "+"%.4f" % rf.oob_score_)

predicted_B = rf.predict(X_test)

print "Original B values:   " + str(y_B_test.values)
print "Predicted B values:  " + str(predicted_B)
print ("Error: " + str(np.mean(y_B_test.values != predicted_B)))



importances = rf.feature_importances_
indices = np.argsort(importances)
feature_cols = feature_cols[indices]
feat_rfc_B = pd.DataFrame(feature_cols)
feat_rfc_B.sort_index(ascending=False,inplace=True)



rf.fit(X_train, y_C_train)
print "Random Forest Classifier For C"
print("oob Score: "+"%.4f" % rf.oob_score_)

predicted_C = rf.predict(X_test)

print "Original C values:   " + str(y_C_test.values)
print "Predicted C values:  " + str(predicted_C)
print ("Error: " + str(np.mean(y_C_test.values != predicted_C)))


importances = rf.feature_importances_
indices = np.argsort(importances)
feature_cols = feature_cols[indices]
feat_rfc_C = pd.DataFrame(feature_cols)
feat_rfc_C.sort_index(ascending=False,inplace=True)



rf.fit(X_train, y_D_train)
print "Random Forest Classifier For D"
print("oob Score: "+"%.4f" % rf.oob_score_)

predicted_D = rf.predict(X_test)

print "Original D values:   " + str(y_D_test.values)
print "Predicted D values:  "  + str(predicted_D)
print ("Error: " + str(np.mean(y_D_test.values != predicted_D)))




importances = rf.feature_importances_
indices = np.argsort(importances)
feature_cols = feature_cols[indices]
feat_rfc_D = pd.DataFrame(feature_cols)
feat_rfc_D.sort_index(ascending=False,inplace=True)





rf.fit(X_train, y_E_train)
print "Random Forest Classifier For E"
print("oob Score: "+"%.4f" % rf.oob_score_)

predicted_E = rf.predict(X_test)

print "Original E values:   " + str(y_E_test.values)
print "Predicted E values:  "  + str(predicted_E)
print ("Error: " + str(np.mean(y_E_test.values != predicted_E)))



importances = rf.feature_importances_
indices = np.argsort(importances)
feature_cols = feature_cols[indices]
feat_rfc_E = pd.DataFrame(feature_cols)
feat_rfc_E.sort_index(ascending=False,inplace=True)




rf.fit(X_train, y_F_train)
print "Random Forest Classifier For F"
print("oob Score: "+"%.4f" % rf.oob_score_)

predicted_F = rf.predict(X_test)

print "Original F values:   " + str(y_F_test.values)
print "Predicted F values:  "  + str(predicted_F)
print ("Error: " + str(np.mean(y_F_test.values != predicted_F)))




importances = rf.feature_importances_
indices = np.argsort(importances)
feature_cols = feature_cols[indices]
feat_rfc_F = pd.DataFrame(feature_cols)
feat_rfc_F.sort_index(ascending=False,inplace=True)

del feature_cols, features, importances, indices



print("PREDICTION OF CHANGE ACCORDING TO NEURAL NETWORK TRAINING")


# Neural Network

from keras.models import Sequential # intitialize the ANN
from keras.layers import Dense    # create layers

model = Sequential()

model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 66))
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_A_train, batch_size = 32, epochs = 4000)

y_pred_A = model.predict(X_test)
y_pred_A = np.round(y_pred_A).astype(int)

y_pred_A = y_pred_A.ravel()

print "Neural Network Trained for A"
print("Values of A:             " + str(y_A_test.values))
print("Predicted values of A:   " + str(y_pred_A))
print("Error : " + str(np.mean(y_A_test.values != y_pred_A)))





model = Sequential()

model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 66))
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])


model.fit(X_train,y_B_train, batch_size=32, epochs = 4000)

y_pred_B = model.predict(X_test)
y_pred_B = np.round(y_pred_B).astype(int)

y_pred_B = y_pred_B.ravel()

print "Neural Network Trained for B"
print("Values of B:             " + str(y_B_test.values))
print("Predicted values of B:   " + str(y_pred_B))
print("Error : " + str(np.mean(y_B_test.values != y_pred_B)))




model = Sequential()

model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 66))
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])


model.fit(X_train,y_C_train, batch_size=32, epochs = 4000)

y_pred_C = model.predict(X_test)
y_pred_C = np.round(y_pred_C).astype(int)

y_pred_C = y_pred_C.ravel()

print "Neural Network Trained for C"
print("Values of C:             " + str(y_C_test.values))
print("Predicted values of C:   " + str(y_pred_C))
print("Error : " + str(np.mean(y_C_test.values != y_pred_C)))




model = Sequential()

model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 66))
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])


model.fit(X_train,y_D_train, batch_size=32, epochs = 4000)

y_pred_D = model.predict(X_test)
y_pred_D = np.round(y_pred_D).astype(int)

y_pred_D = y_pred_D.ravel()

print "Neural Network Trained for D"
print("Values of D:             " + str(y_D_test.values))
print("Predicted values of D:   " + str(y_pred_D))
print("Error : " + str(np.mean(y_D_test.values != y_pred_D)))





model = Sequential()

model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 66))
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])


model.fit(X_train,y_E_train, batch_size=32, epochs = 4000)

y_pred_E = model.predict(X_test)
y_pred_E = np.round(y_pred_E).astype(int)

y_pred_E = y_pred_E.ravel()

print "Neural Network Trained for E"
print("Values of E:             " + str(y_E_test.values))
print("Predicted values of E:   " + str(y_pred_E))
print("Error : " + str(np.mean(y_E_test.values != y_pred_E)))





model = Sequential()

model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 66))
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])


model.fit(X_train,y_F_train, batch_size=32, epochs = 4000)

y_pred_F = model.predict(X_test)
y_pred_F = np.round(y_pred_F).astype(int)

y_pred_F = y_pred_F.ravel()

print "Neural Network Trained for F"
print("Values of F:             " + str(y_F_test.values))
print("Predicted values of F:   " + str(y_pred_F))
print("Error : " + str(np.mean(y_F_test.values != y_pred_F)))

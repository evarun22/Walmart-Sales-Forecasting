from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
sns.set_style("whitegrid")
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pandas import datetime
import h5py
import warnings
warnings.filterwarnings('ignore')
    
    
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('login.html')

@app.route('/index', methods=['GET','POST'])
def index():
    user=request.form['un']
    pas=request.form['pw']
    cr=pd.read_excel('cred.xlsx')
    un=np.asarray(cr['Username']).tolist()
    pw=np.asarray(cr['Password']).tolist()
    cred = dict(zip(un, pw))
    if user in un:
        if(cred[user]==pas):
            return render_template('index.html')
        else:
            k=1
            return render_template('login.html',k=k)
        
    else:
        k=1
        return render_template('login.html',k=k)


@app.route('/data_viz')
def data_viz():
    return render_template("data_viz.html")

@app.route('/file_upload')
def file_upload():
    return render_template("file_upload.html")

@app.route('/upload_printed', methods=['GET','POST'])
def upload_printed():
    abc=request.files['printed_doc']
    
    

    test1=pd.read_csv(abc)
    test=test1
    
    train=pd.read_csv('train.csv')
    #test=pd.read_csv('test.csv')
    store=pd.read_csv('stores.csv')
    feature=pd.read_csv('features.csv')
    
    print("\nEXPLORING store.csv")
    print("\n",store.head().append(store.tail()),"\n")
    print("Structure of Store:\n",store.shape, "\n")
    print("Number of missing values:\n",store.isnull().sum().sort_values(ascending=False),"\n")
   
    print("\nEXPLORING feature.csv")
    print(feature.head().append(feature.tail()),"\n")
    print("Structure of Feature: ",feature.shape,"\n")
    print("Summary Statistic:\n",feature.describe(),"\n")
    print("Number of missing values:\n",feature.isnull().sum().sort_values(ascending=False),"\n")
    
    print("\nFINDING OUT THE MISSING PERCENTAGE OF DATA ACROSS EACH FEATURE")
    feature_percent_missing = feature.isnull().sum()*100/len(feature)
    feature_data_type = feature.dtypes
    feature_summary = pd.DataFrame({"Percent_missing": feature_percent_missing.round(2), 
                               "Datatypes": feature_data_type})
    print('\n',feature_summary)
    
    print("\nEXPLORING train.csv")    
    print(train.head().append(train.tail()),"\n")
    print("Structure of train:\n",train.shape,"\n")
    print("Summary Statistic:\n",train.describe(),"\n")
    print("Number of missing values:\n",train.isnull().sum().sort_values(ascending=False),"\n")
    
    print("\nEXPLORING test.csv")    
    print(test.head().append(test.tail()),"\n")
    print("Structure of test:\n",test.shape,"\n")
    print("Summary Statistic:\n",test.describe(),"\n")
    print("Number of missing values:\n",test.isnull().sum().sort_values(ascending=False),"\n")
    
    print('\nJOINING TABLES:')
    combined_train = pd.merge(train, store, how="left", on="Store")
    combined_test = pd.merge(test, store, how="left", on="Store")
    
    combined_train = pd.merge(combined_train, feature, how = "inner", on=["Store","Date"])
    combined_test = pd.merge(combined_test, feature, how = "inner", on=["Store","Date"])

    combined_train = combined_train.drop(["IsHoliday_y"], axis=1)
    combined_test = combined_test.drop(["IsHoliday_y"], axis=1)
    
    print(combined_train.head(),"\n", combined_train.shape,"\n")
    print(combined_test.head(),"\n", combined_test.shape,"\n")
    
    print(combined_train.describe())
    print(combined_test.describe())
    
    print('\nDATA PREPROCESSING:')
    print('\nREPLACING MISSING VALUES BY 0')
    print('\nCHECKING FOR THE TOTAL NUMBER OF MISSING VALUES IN combined_train AND combined_test AND THEN REPLACING THEM WITH 0')
    
    print(combined_test.isnull().sum())
    print(combined_train.isnull().sum())
    
    processed_train = combined_train.fillna(0)
    processed_test = combined_test.fillna(0)
    
    print('\nREPLACING NEGATIVE MARKDOWN EVENTS BY 0 IN processed_train AND processed_test')
    
    processed_train.loc[processed_train['Weekly_Sales'] < 0.0,'Weekly_Sales'] = 0.0
    processed_train.loc[processed_train['MarkDown2'] < 0.0,'MarkDown2'] = 0.0
    processed_train.loc[processed_train['MarkDown3'] < 0.0,'MarkDown3'] = 0.0
    
    print('\n',processed_train.describe())
    
    processed_test.loc[processed_test['MarkDown1'] < 0.0,'MarkDown1'] = 0.0
    processed_test.loc[processed_test['MarkDown2'] < 0.0,'MarkDown2'] = 0.0
    processed_test.loc[processed_test['MarkDown3'] < 0.0,'MarkDown3'] = 0.0
    processed_test.loc[processed_test['MarkDown5'] < 0.0,'MarkDown5'] = 0.0
    
    print('\n',processed_test.describe())
    
    print('\nPERFORMING ONE HOT ENCODING FOR CATEGORICAL DATA AND BOOLEAN DATA:')
    
    print('\n',processed_train.dtypes, processed_test.dtypes)
    
    cat_col = ['IsHoliday_x','Type']
    for col in cat_col:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(processed_train[col].values.astype('str'))
        processed_train[col] = lbl.transform(processed_train[col].values.astype('str'))
    for col in cat_col:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(processed_test[col].values.astype('str'))
        processed_test[col] = lbl.transform(processed_test[col].values.astype('str'))
        
    processed_test.to_csv("Processed_data/processed_test.csv", index=False)
    print('\n',processed_test.head())
    
    processed_train = processed_train[['Store', 'Dept', 'Date', 'Unemployment', 'IsHoliday_x', 'Type', 'Size',
       'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3',
       'MarkDown4', 'MarkDown5', 'CPI', 'Weekly_Sales']]
    processed_train.to_csv("Processed_data/processed_train.csv", index=False)
    print('\n',processed_train.head())
    
    '''print('\nVISUALIZATION OF HISTORIC DATA:')
    
    store['Type'].value_counts().plot(kind='bar')
    plt.title('Total number of each type of stores')
    plt.xlabel('Type')
    plt.ylabel('Number of Stores')
    plt.show()
    
    a=sns.catplot(x="Type", y="Size", data=store);
    a.fig.suptitle('Sizes of each type of store')
    
    a=train[['Store', 'Dept']].drop_duplicates()
    a.plot(kind='scatter', x='Store',y='Dept')
    plt.title('Departments across every store')
    
    print('\nPLOTTING CORRELATION HEATMAP:')
    corr=processed_train.corr()
    sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns)
    
    print('\nPLOTTING CORRELATION MATRIX:')
    cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

    def magnify():
        return [dict(selector="th",props=[("font-size", "7pt")]),
                dict(selector="td",props=[('padding', "0em 0em")]),
                dict(selector="th:hover",props=[("font-size", "12pt")]),
                dict(selector="tr:hover td:hover",props=[('max-width', '200px'),('font-size', '12pt')])]

    corr.style.background_gradient(cmap, axis=1)\
        .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
        .set_caption("Hover to magify")\
        .set_precision(2)\
        .set_table_styles(magnify())'''
    
    dfabc=processed_train[['Date','Store','Dept','IsHoliday_x','Unemployment','Fuel_Price','Temperature','Type','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','CPI','Weekly_Sales']]
    
    dfabc["MarkDownValue"] = dfabc["MarkDown1"].add(dfabc["MarkDown2"])
    dfabc["MarkDownValue"] = dfabc["MarkDownValue"].add(dfabc["MarkDown3"])
    dfabc["MarkDownValue"] = dfabc["MarkDownValue"].add(dfabc["MarkDown4"])
    dfabc["MarkDownValue"] = dfabc["MarkDownValue"].add(dfabc["MarkDown5"])
    
    dfabc = dfabc[dfabc.MarkDownValue != 0.0]
    
    dfdef=processed_test[['Date','Store','Dept','IsHoliday_x','Type','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','CPI']]

    dfdef["MarkDownValue"] = dfdef["MarkDown1"].add(dfdef["MarkDown2"])
    dfdef["MarkDownValue"] = dfdef["MarkDownValue"].add(dfdef["MarkDown3"])
    dfdef["MarkDownValue"] = dfdef["MarkDownValue"].add(dfdef["MarkDown4"])
    dfdef["MarkDownValue"] = dfdef["MarkDownValue"].add(dfdef["MarkDown5"])

    dfdef = dfdef[dfdef.MarkDownValue != 0.0]
    
    dfx=dfabc
    dfx=pd.get_dummies(dfx, columns=['Dept','Store','Type'])
    dfx['Day']=dfx['Date'].str[0:2]
    dfx['Month']=dfx['Date'].str[3:5]
    dfx['Year']=dfx['Date'].str[6:10]
    
    dfx['Day']=pd.to_numeric(dfx['Day'])
    dfx['Month']=pd.to_numeric(dfx['Month'])
    dfx['Year']=pd.to_numeric(dfx['Year'])
    
    dftest=dfdef
    dftest=pd.get_dummies(dftest, columns=['Dept','Store','Type'])
    dftest['Day']=dftest['Date'].str[0:2]
    dftest['Month']=dftest['Date'].str[3:5]
    dftest['Year']=dftest['Date'].str[6:10]
    
    dftest['Day']=pd.to_numeric(dftest['Day'])
    dftest['Month']=pd.to_numeric(dftest['Month'])
    dftest['Year']=pd.to_numeric(dftest['Year'])
    #print(dfx.head())

    
    x=dfx[[#'Unemployment',
            'IsHoliday_x',
            #'Size','Temperature','Fuel_Price',
            'MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5',
            #'CPI','Weekly_Sales',
            'Dept_1','Dept_2','Dept_3','Dept_4','Dept_5',
            'Dept_6','Dept_7','Dept_8','Dept_9','Dept_10',
            'Dept_11','Dept_12','Dept_13','Dept_14','Dept_16',
            'Dept_17','Dept_18','Dept_19','Dept_20','Dept_21',
            'Dept_22','Dept_23','Dept_24','Dept_25','Dept_26',
            'Dept_27','Dept_28','Dept_29','Dept_30','Dept_31',
            'Dept_32','Dept_33','Dept_34','Dept_35','Dept_36',
            'Dept_37','Dept_38','Dept_39','Dept_40','Dept_41',
            'Dept_42','Dept_43','Dept_44','Dept_45','Dept_46',
            'Store_1','Store_2','Store_3','Store_4','Store_5',
            'Store_6','Store_7','Store_8','Store_9','Store_10',
            'Store_11','Store_12','Store_13','Store_14','Store_15',
            'Store_16','Store_17','Store_18','Store_19','Store_20',
            'Store_21','Store_22','Store_23','Store_24','Store_25',
            'Store_26','Store_27','Store_28','Store_29','Store_30',
            'Store_31','Store_32','Store_33','Store_34','Store_35',
            'Store_36','Store_37','Store_38','Store_39','Store_40',
            'Store_41','Store_42','Store_43','Store_44','Store_45',
            'Type_0','Type_1','Type_2','Day','Month','Year']]
    y=dfx[['Weekly_Sales']]
    #yhat
    #z.shape


    rf=RandomForestRegressor()
    
    xtest=dftest[[#'Unemployment',
            'IsHoliday_x',#'Size','Temperature','Fuel_Price',
            'MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5',
            #'CPI','Weekly_Sales',
            'Dept_1','Dept_2','Dept_3','Dept_4','Dept_5',
            'Dept_6','Dept_7','Dept_8','Dept_9','Dept_10',
            'Dept_11','Dept_12','Dept_13','Dept_14','Dept_16',
            'Dept_17','Dept_18','Dept_19','Dept_20','Dept_21',
            'Dept_22','Dept_23','Dept_24','Dept_25','Dept_26',
            'Dept_27','Dept_28','Dept_29','Dept_30','Dept_31',
            'Dept_32','Dept_33','Dept_34','Dept_35','Dept_36',
            'Dept_37','Dept_38','Dept_39','Dept_40','Dept_41',
            'Dept_42','Dept_43','Dept_44','Dept_45','Dept_46',
            'Store_1','Store_2','Store_3','Store_4','Store_5',
            'Store_6','Store_7','Store_8','Store_9','Store_10',
            'Store_11','Store_12','Store_13','Store_14','Store_15',
            'Store_16','Store_17','Store_18','Store_19','Store_20',
            'Store_21','Store_22','Store_23','Store_24','Store_25',
            'Store_26','Store_27','Store_28','Store_29','Store_30',
            'Store_31','Store_32','Store_33','Store_34','Store_35',
            'Store_36','Store_37','Store_38','Store_39','Store_40',
            'Store_41','Store_42','Store_43','Store_44','Store_45',
            'Type_0','Type_1','Type_2','Day','Month','Year']]
    
    rf.fit(x,y)
    yhat=rf.predict(xtest)
    processed_test['Predicted Sales']=yhat
    processed_test.to_excel('result.xlsx')
    
    #test1.to_excel('result.xlsx')
    
    return render_template("upload_printed.html")
                           #,tables=[disp.to_html(classes='data')], titles=disp.columns.values[-1:])

@app.route('/attribute_entry')
def attribute_entry():
    return render_template('attribute_entry.html')


if __name__ == '__main__':
    app.run()

import json
import pandas as pd
from scipy import stats
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import xlsxwriter

from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import r2_score#R square

from sklearn.model_selection import train_test_split,cross_val_score	#划分数据 交叉验证

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
import sklearn.gaussian_process  # GPR
from sklearn import tree  # DTR
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, ExpSineSquared, RBF, ConstantKernel
from sklearn.gaussian_process.kernels import Exponentiation, RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.svm import SVR  # SVR
from sklearn.kernel_ridge import KernelRidge  #KRR

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))

def R2(y_true, y_pred):
    return r2_score(y_true,y_pred)

def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred))

def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

with open('Features/FeaturesInPressureMap_train.json') as f:
    load_dict_train = json.load(f)

with open('Features/ThreeDFeatures_PressMap.json') as f:
    tf = json.load(f)

with open('Features/FeaturesInPressureMap_test.json') as f:
    load_dict_test = json.load(f)

# with open('/home/benkesheng/RGZNYL/pyx/BMIfeature/ThreeDFeatures_RGB_test.json') as f:
#     tf_test = json.load(f)

def DataProcess(load_dict, tf):
    Features_ALL = []
    for key in load_dict.keys():
        features = []
        features.append(load_dict[key]['deep features'])
        features.append(load_dict[key]['Range'])
        features.append(load_dict[key]['Mean'])
        features.append(load_dict[key]['Variance'])
        features.append(load_dict[key]['Skewness'])
        features.append(load_dict[key]['WTR'])
        features.append(load_dict[key]['WSR'])
        features.append(load_dict[key]['HpHdR'])
        features.append(load_dict[key]['WHdR'])
        features.append(load_dict[key]['H2W'])
        features.append(load_dict[key]['WHpR'])

        features.append(tf[key]['maxLength'])
        features.append(tf[key]['maxWidth'])
        features.append(tf[key]['eigenvalue1'])
        features.append(tf[key]['eigenvalue2'])
        features.append(tf[key]['eigenvalue3'])
        features.append(tf[key]['sphericity'])
        features.append(tf[key]['linearity'])
        features.append(tf[key]['compactness'])
        features.append(tf[key]['kurtosis'])

        features.append(load_dict[key]['BMI'])

        Features_ALL.append(features)

    ALL = pd.DataFrame(Features_ALL, columns=['deep features',
                                            'Range', 'Mean', 'Variance', 'Skewness',
                                            'WTR','WSR', 'HpHdR', 'WHdR', 'H2W', 'WHpR',
                                            'maxLength', 'maxWidth', 'eigenvalue1',
                                            'eigenvalue2', 'eigenvalue3', 'sphericity',
                                            'linearity', 'compactness', 'kurtosis', 'BMI'])

    All_Features = ALL.iloc[:, :-1]
    All_Features = All_Features.replace([np.inf, -np.inf], np.nan)
    All_Features = All_Features.replace(np.nan, 0)
    BMI = ALL.iloc[:, -1]
    return All_Features.values, BMI.values

# #Code waiting for modification
# def DataProcess(state):
#     data = Guo('BMIfeature/')
#     feas = data.json_data()
#     df_data, sf_data, bf_data, bmi_data, sex= {}, {}, {}, {}, {}
#     lenth = len(feas)
#     for idx in range(0,lenth-1):
#         df_data[idx] = np.asarray(feas[idx][0])
#         sf_data[idx] = np.asarray(feas[idx][1:9])
#         bf_data[idx] = np.asarray(feas[idx][9:16])
#         bmi_data[idx] = np.asarray(feas[idx][16])
#         sex[idx] = np.asarray(feas[idx][17])
#     df_train, df_test, bmi_train_df, bmi_test_df = train_test_split(df_data, bmi_data, test_size=1/10,random_state=state)
#     sf_train, sf_test, bmi_train_sf, bmi_test_sf = train_test_split(sf_data, bmi_data, test_size=1/10,random_state=state)
#     bf_train, bf_test, bmi_train_bf, bmi_test_bf = train_test_split(bf_data, bmi_data, test_size=1/10,random_state=state)
#     return df_train, df_test, bmi_train_df, bmi_test_df, sf_train, sf_test, bmi_train_sf, bmi_test_sf, bf_train, bf_test, bmi_train_bf, bmi_test_bf

All_Features_train, BMI_train = DataProcess(load_dict_train, tf)
All_Features_test, BMI_test = DataProcess(load_dict_test, tf)

df_train, df_test, bf_train, bf_test, sf_train, sf_test, tf_train, tf_test, = [], [], [], [], [], [], [], []
for per in range(0, len(All_Features_train)):
    df_train.append(np.asarray(All_Features_train[per][0]))
    sf_train.append(np.asarray(All_Features_train[per][1:5]))
    bf_train.append(np.asarray(All_Features_train[per][5:11]))
    tf_train.append(np.asarray(All_Features_train[per][11:20]))

for per in range(0, len(All_Features_test)):
    df_test.append(np.asarray(All_Features_test[per][0]))
    sf_test.append(np.asarray(All_Features_test[per][1:5]))
    bf_test.append(np.asarray(All_Features_test[per][5:11]))
    tf_test.append(np.asarray(All_Features_test[per][11:20]))

# train_data = tf_train
# test_data = tf_test
train_data = np.hstack((sf_train, tf_train, bf_train, df_train))
test_data = np.hstack((sf_test, tf_test, bf_test, df_test))
#print(np.shape(test_data))

def Regression():
    # 初始化
    lr  = LinearRegression()
    svr = SVR(kernel='linear')
    dtr = tree.DecisionTreeRegressor()
    bte = BaggingRegressor()
    # KN = Exponentiation(RationalQuadratic(), exponent=2)
    # gpr = sklearn.gaussian_process.GaussianProcessRegressor(kernel=KN, alpha=1e-3)
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


    regressors = [bte, KRR]
    y_pred = [{}, {}]
    reg_name = ['BTE', 'KRR']

    r2, rmse, mae, mape = {}, {}, {}, {}
    for i, reg in enumerate(regressors):
        for k in range(0,10):
            reg.fit(train_data, BMI_train)
            y_pred[i][k] = reg.predict(test_data)
            r2[k] = R2(BMI_test, y_pred[i][k])
            rmse[k] = RMSE(BMI_test, y_pred[i][k])
            mae[k] = MAE(BMI_test, y_pred[i][k])
            mape[k] = MAPE(BMI_test, y_pred[i][k])
        print(reg_name[i], ': R2: ', sum(r2.values())/len(r2), ' RMSE: ', sum(rmse.values())/len(rmse),
                           ' MAE: ',sum(mae.values())/len(mae), ' MAPE ', sum(mape.values())/len(mape))
    # y_pred = {}
    # bte.fit(train_data, BMI_train)
    # y_pred = bte.predict(test_data)
    # data = np.concatenate((y_pred, BMI_test),axis=0)
    # #data = np.vstack(y_pred, BMI_test)
    # data = pd.DataFrame(data)

    # writer = pd.ExcelWriter('prediction_sf.xlsx', engine = "xlsxwriter")
    # data.to_excel(writer, 'sf', float_format='%.5f')
    # writer.save()
    # writer.close

Regression()





# ライブラリのインポート

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# データセットのインポート

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values

print(X)



# カテゴリ変数のエンコーディング

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')

X = np.array(ct.fit_transform(X))

print(X)



# ダミー変数トラップを避けるためのコード

X = X[:, 1:]



# 訓練用とテスト用へのデータセットの分割

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# 訓練用データセットを使った重回帰モデルの訓練

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



# テスト用データを使った結果の予測

y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))



# 変数減少法を使った変数の抽出

import statsmodels.api as sm

X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

X_opt = X[:, [0, 1, 2, 3, 4, 5]]

X_opt = X_opt.astype(np.float64)

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]

X_opt = X_opt.astype(np.float64)

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]

X_opt = X_opt.astype(np.float64)

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()X_opt = X[:, [0, 3, 5]]

X_opt = X_opt.astype(np.float64)

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()

X_opt = X[:, [0, 3]]

X_opt = X_opt.astype(np.float64)

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()

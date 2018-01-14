import pandas as pd

# ##载入数据集并观察
df_train = pd.read_csv('input/train.csv')
df_test = pd.read_csv('input/test.csv')
print(df_train.columns)
print(df_train.head(5))

# 1，样本量小；
# 2，训练集与测试集的数量基本相当
print("Training set size is %d"%df_train.shape[0])
print("Testing set size is %d"%df_test.shape[0])

# 异常值情况
print('Count the na entries in each train set columns')
print(df_train.isnull().sum())
print('Count the na entries in each test set columns')
print(df_test.isnull().sum())
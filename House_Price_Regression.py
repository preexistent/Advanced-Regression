
# coding: utf-8

# # 使用回归方法，预测房产价格
# 
# 这个工程中，将应用回归树、线性回归、xgboost回归等机器学习的方法，对房产的价格进行预测。
# 具体的步骤简述如下：
# 
# 1. 载入并观察数据
# 2. categorical类型特征工程
# 3. numerical类特征工程
# 4. 处理异常值
# 5. 构造机器学习模型
# 6. 输入处理后数据，并交叉验证，选择最优模型
# 7. 按照evaluation规定，对模型的准确度进行评价

# In[32]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

get_ipython().magic('matplotlib inline')


# ## 载入数据集并观察

# In[33]:


df_train = pd.read_csv('input/train.csv')
df_test = pd.read_csv('input/test.csv')
print(df_train.columns)
df_train.head(5)


# ### 数据集大小

# In[25]:


# 1，样本量小；
# 2，训练集与测试集的数量基本相当
print("Training set size is %d"%df_train.shape[0])
print("Testing set size is %d"%df_test.shape[0])


# ### 异常值情况

# In[11]:


print('Count the na entries in each train set columns')
print(df_train.isnull().sum())
print('Count the na entries in each test set columns')
print(df_test.isnull().sum())


# 可看出：
# 1. 有些列存在严重的na情况；
# 2. 一些缺失的列存在强关联关系；
# 3. 有些列基本在所有样本中缺失。

# ### 训练集与测试集在异常值方面的不同

# 下述的特征，在训练集及测试集中存在不同的异常情况。在异常处理中，需要进行特殊的处理

# In[34]:


train_index = df_train.iloc[:, :-1].isnull().sum()>0
test_index = df_test.isnull().sum()>0
diff = np.logical_xor(train_index, test_index)

diff_index = diff[diff==True].index
print(diff_index)


# ## 特征选取

# ### 分辨categorical与numerical类型特征
# 对于categorical与numerical特征的分辨，主要目的在于之后对不同类型的特征使用不同的特征工程方法。

# In[35]:


quantitative = []
qualitative = []

quantitative = [f for f in df_train.columns if df_train.dtypes[f] != 'object']
quantitative.remove('Id')
quantitative.remove('SalePrice')
qualitative = [f for f in df_train.columns if df_train.dtypes[f] == 'object']

print('numerical features include:')
print(quantitative)
print('categorical features include:')
print(qualitative)


# ### categorical特征工程

# 这里说明一下，特征工程中，主要可分为两个步骤：
# * 研究特征与label之间的关系，而这关系主要是由训练集中求得。
# * 对特征数据进行处理，这部分工作就要对训练集及测试集同时进行。
# 
# 下面对categorical类特征进行特征工程，主要步骤包括：
# 1. 使用boxplot直观理解关系；
# 2. anova方法研究具体关系；
# 3. 编码
# 
# 这里分析一下所得的这些图。与上个文档中对于boxplot的使用方法类似，这里也是使用boxplot方法直观地研究categorical特征与回归值的关系。但本工程中对于特征值的着重点不同。具体来说： 1，这个工程中，主要注意的是图中，每个特征值对应的中值与其他特征值的变化情况。变化大，就说明该特征的不同取值对于回归值有影响；而在后边的anova分析中，也是用到了类似的概念； 2，该分析中并没有强调如第一个工程里，对于线性因素的挖掘。直观讲，这样也更加合理。因为categorical特征取值之间并没有数学意义，所以谈论线性也就无根据。

# In[28]:


for f in qualitative:
    df_train[f] = df_train[f].astype('category')
    df_test[f] = df_test[f].astype('category')
    if df_train[f].isnull().any():
        df_train[f] = df_train[f].cat.add_categories(['MISSING'])
        df_train[f] = df_train[f].fillna('MISSING')
    if df_test[f].isnull().any():
        df_test[f] = df_test[f].cat.add_categories(['MISSING'])
        df_test[f] = df_test[f].fillna('MISSING')

def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x = plt.xticks(rotation=90)
f = pd.melt(df_train, id_vars=['SalePrice'], value_vars=qualitative)
g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(boxplot, 'value', 'SalePrice')


# 下面使用anova，即方差分析方法对categorical数据的统计特性进行分析。
# 
# 这里要补充一下对于anova的知识。
# anova中文翻译为：方差分析，具体的定义可以百度或者wiki。 在这里，使用anova主要是检验变量是否具有统计意义，或者说特征值的改变是否能够显著改变回归值。 f_oneway中，fval为得出的f值，而pval则是在F分布中，对应f值的概率。 当P>0.05时，我们就称特征不具备统计意义，或者说没有明显影响回归值。

# In[11]:


def anova(frame):
    anv = pd.DataFrame()
    anv['feature'] = qualitative
    pvals = []
    for c in qualitative:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c]==cls]['SalePrice'].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

a = anova(df_train)
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)

这两组书的意义是，如果上图中的数值高于3，我们就可以说说，特征对于label的影响是显著的。在这里，按照各特征取值下，对应的saleprice的中值进行排序； 并按照该序列对categorical特征进行转码。 这样，从一定程度上，也增加了categorical特征与回归值之间的相关性。
# #### categorical特征编码

# In[37]:


def encode(frame, feature):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['spmean']= frame[[feature,'SalePrice']].groupby(feature).mean()['SalePrice']
#     print(frame[[feature,'SalePrice']].groupby(feature).mean())
    ordering = ordering.sort_values('spmean')
    ordering['order'] = range(1, ordering.shape[0]+1)
#     print(ordering)
    ordering = ordering.order.to_dict()
#     print(ordering)
#     print('-------------------')
    
    for val, ord in ordering.items():
        frame.loc[frame[feature]==val, feature+'_E'] = ord
        
    return ordering

qual_encoded = []
train_ordering = {}
for q in qualitative:
    train_ordering[q] = encode(df_train, q)
    qual_encoded.append(q+'_E')
print(qual_encoded)


# In[38]:


# 测试集categorical特征编码

for index, di in train_ordering.items():
    for val, ord in di.items():
        df_test.loc[df_test[index]==val, index+'_E'] = ord
 
# 编码后，就可以将训练集及测试集中的categorical类型的变量删除
df_train.drop(qualitative, axis=1, inplace=True)
df_test.drop(qualitative, axis=1, inplace=True)
df_train.head(5)


# ### 相关性分析
# 
# 在categorical特征转化为numerical特征后，就可以对整个特征空间的相关性进行分析了。
# 相关图能够定性地给出变量间的相关关系，实现特征筛选的第一次快速迭代。

# In[18]:


corrmat = df_train.corr(method='spearman')
k = 20
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(25,25))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[19]:


k = 20
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
print(cols)

第一次迭代中，我们先选取相关性最高的10个特征进行分析。这些特征包括：
'OverallQual', 
'Neighborhood_E', 
'GrLivArea', 
'GarageCars',
'ExterQual_E', 
'BsmtQual_E', 
'KitchenQual_E',
'YearBuilt', 
'GarageArea',
'FullBath'

但考虑到一些特征之间存在强相关关系，对这些特征的选取还要讨论。（高于与saleprice的相关性0.05以上）
GarageArea与GarageCars。二者的相关关系很明显，这里只取与saleprice相关性更强的garagecars。
ExterQual_E与OverallQual。二者取OverallQual
yearbuilt与很多属性的相关性都大于saleprice，暂且舍弃。
Fullbatch与GrLivArea。可以想象，groundarea大的情况下，fullbath也会大，两者是存在一定的相关性的。但直观上，bath应该还能够提供其他的信息，所以暂且保留。
补充totalbsmtSF。在这里我要对之前的笔记进行一定的修正。
首先，通过实验确定，更多的特征，即便是相互相关性较强的特征，能够提高回归器的能力。
而减少特征的目的主要存在于两个方面：
1，减少计算量；
2，在给定的样本空间下，过多的特征影响机器学习的效能。
第一条很直观。
第二条我认为，随着特征的剧烈增加，样本空间能够覆盖的特征值组合减少；所以模型无法有效地学习所有特征值组合的概率分布。
这句话隐含的意思，是测试集的特征空间增加，则更有可能出现训练集完全没有学习到的分布，从而降低学习器的能力。

但就本题来讲，因为高相关性的特征数量相较于样本值并不多，所以不需要删除过多特征。
# ## 缺失值及异常值处理
在得出重要的特征后，就可以根据这些选择的特征进行异常值的处理。
之所以将异常值的处理放在numerical特征工程前，是因为特征取值分布，或者log这样的操作，对于异常值是敏感的。
# In[20]:


abn_f_train = [feature for feature in df_train.columns if df_train[feature].isnull().sum()>0]
abn_f_test = [feature for feature in df_test.columns if df_test[feature].isnull().sum()>0]
print(abn_f_train)
print('`````````')
print(abn_f_test)

可以看出，在测试集中，出现了一些训练集中没有出现的缺失值。
为了最大限度的保留特征，需要对下面缺失值进行补充。
# In[21]:


fs = [f for f in abn_f_test if f not in abn_f_train]
print(df_test[fs].isnull().sum())


# In[22]:


df_test.loc[df_test['GarageCars'].isnull(), 'GarageCars'] = 0
df_test.loc[df_test['GarageArea'].isnull(), 'GarageArea'] = 0

从下表可以看出，缺失值之间很多是同一个样本。
# In[23]:


print(df_test.loc[df_test['BsmtFinSF1'].isnull()].index)
print(df_test.loc[df_test['BsmtFinSF1'].isnull()].index)
print(df_test.loc[df_test['BsmtUnfSF'].isnull()].index)
print(df_test.loc[df_test['TotalBsmtSF'].isnull()].index)
print('``````')
print(df_test.loc[df_test['BsmtFullBath'].isnull()].index)
print('``````')
print(df_test.loc[df_test['BsmtHalfBath'].isnull()].index)
print('``````')
print(df_test.loc[df_test['Exterior1st_E'].isnull()].index)
print(df_test.loc[df_test['Exterior2nd_E'].isnull()].index)
print(df_test.loc[df_test['Exterior2nd_E'].isnull()].index)
print('``````')
print(df_test.loc[df_test['KitchenQual_E'].isnull()].index)
print(df_test.loc[df_test['Functional_E'].isnull()].index)
print(df_test.loc[df_test['SaleType_E'].isnull()].index)

根据特征说明，是否具有bsmt可以通过bsmtfintype来查询。
# In[24]:


print(train_ordering)

通过查找特征表，当fintype为1时，表示没有bsmt。所以缺失值这里应该填充0。
# In[25]:


df_test.loc[df_test['BsmtFinSF1'].isnull(), 'BsmtFinSF1'] = 0
df_test.loc[df_test['BsmtFinSF2'].isnull(), 'BsmtFinSF2'] = 0
df_test.loc[df_test['BsmtUnfSF'].isnull(), 'BsmtUnfSF'] = 0
df_test.loc[df_test['TotalBsmtSF'].isnull(), 'TotalBsmtSF'] = 0


# In[26]:


df_test.loc[df_test['BsmtFullBath'].isnull(), 'BsmtFullBath'] = 0
df_test.loc[df_test['BsmtHalfBath'].isnull(), 'BsmtHalfBath'] = 0

Kitchen_qual与exter的情况类似，与其相关性强的特征为Exter。
根据exter的信息，KitchenQual_E的异常值使用当exter为2时，Kitchen的最多值2作为填充。
# In[27]:


KQ_dis = df_test[df_test['ExterQual_E']==2]
KQ_dis.KitchenQual_E.describe()


# In[28]:


df_test.loc[df_test['KitchenQual_E'].isnull(), 'KitchenQual_E'] = 2

剩下的缺失值，由于能够得到的信息较少，所以，这里暂且对其填充为该特征的最多值。
# In[29]:


df_test.loc[df_test['MSZoning_E'].isnull(), 'MSZoning_E'] = df_test['MSZoning_E'].value_counts().idxmax()
df_test.loc[df_test['Utilities_E'].isnull(), 'Utilities_E'] = df_test['Utilities_E'].value_counts().idxmax()
df_test.loc[df_test['Exterior1st_E'].isnull(), 'Exterior1st_E'] = df_test['Exterior1st_E'].value_counts().idxmax()
df_test.loc[df_test['Exterior2nd_E'].isnull(), 'Exterior2nd_E'] = df_test['Exterior2nd_E'].value_counts().idxmax()
df_test.loc[df_test['Functional_E'].isnull(), 'Functional_E'] = df_test['Functional_E'].value_counts().idxmax()
df_test.loc[df_test['SaleType_E'].isnull(), 'SaleType_E'] = df_test['SaleType_E'].value_counts().idxmax()

剩下的三个存在缺失值的特征，是df_train与df_test中共存的。
# In[30]:


fs = [f for f in abn_f_test if f in abn_f_train]
print(df_train[fs].isnull().sum())
print('```````')
print(df_test[fs].isnull().sum())


# In[31]:


print(df_train.loc[df_train['MasVnrArea'].isnull(), 'MasVnrType_E'])
print(train_ordering['MasVnrType'])


# In[32]:


df_test.loc[df_test['MasVnrArea'].isnull(), 'MasVnrArea'] = 0
df_train.loc[df_train['MasVnrArea'].isnull(), 'MasVnrArea'] = 0


# In[33]:


print(df_train.loc[df_train['GarageYrBlt'].isnull(), 'GarageFinish_E'])
print(train_ordering['GarageFinish'])


# In[34]:


df_train.loc[df_train['GarageYrBlt'].isnull(), 'GarageYrBlt'] = 0
df_test.loc[df_test['GarageYrBlt'].isnull(), 'GarageYrBlt'] = 0


# In[35]:


print(df_train.loc[df_train['LotFrontage'].isnull(), 'Street_E'])
print(train_ordering['Street'])
print(df_train.loc[df_train['Street_E']==2.0, 'LotFrontage'])


# In[36]:


df_train.loc[df_train['LotFrontage'].isnull(), 'LotFrontage'] = np.mean(df_train.loc[df_train['Street_E']==2.0, 'LotFrontage'])
df_test.loc[df_test['LotFrontage'].isnull(), 'LotFrontage'] = np.mean(df_test.loc[df_test['Street_E']==2.0, 'LotFrontage'])


# In[37]:


print(np.sum(df_train.isnull()))
print('``````')
print(np.sum(df_test.isnull()))


# In[38]:


# Looking for outliers, as indicated in https://ww2.amstat.org/publications/jse/v19n3/decock.pdf
plt.scatter(df_train.GrLivArea, df_train.SalePrice, c = "blue", marker = "s")
plt.title("Looking for outliers")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.show()

df_train = df_train[df_train.GrLivArea < 4000]


# In[39]:


df_train.replace([-np.inf],0)
df_test.replace([-np.inf],0)
df_train.replace([np.nan],0)
df_test.replace([np.nan],0)


# ## Numerical特征工程
按照相关性，我们筛选出了一些重要的特征。
这些特征中，有些是categorical特征编码而来的。这种情况下，特征的各种取值之间是不存在分布的关系的；
其余特征是原本为numerical的特征。这些特征，包括label saleprice，根据中心极限定理，我们希望他们的分布按照正态分布。
而numerical特征工程的工作，就是观察特征取值分布，并对不符合模型所要求的特征，进行分布拟合。下面，在将特征输入到模型中前，要对数据进行检验，主要包括四个方面：
1，正态性；
2，同质性。该属性是说，特征数据在整个label上都具备同样量级的变化性。
3，线性；
4，相关性错误。
这部分的内容还学要我去特殊看一下Hair的书，我已经下来。
# ### 正态性

# In[40]:


from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

从分布中清楚看到，label的分布并不符合正态。
分布具有左偏斜。
原文中，作者在此处使用了log的方法，将label变换为正态分布。这里需要研究的几个特征包括：
1，saleprice
2, GrLivArea
3, TotalBsmtSF

此外，还有一些离散的numerical特征，这里也试图进行处理。
1，OverallQual
2，GarageCars
# In[41]:


sns.distplot(df_train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


# In[42]:


df_train.SalePrice = np.log1p(df_train.SalePrice)


# In[43]:


skewness = df_train[quantitative].apply(lambda x: stats.skew(x))
skewness = skewness[skewness > 0.5]
feature_skewed = skewness.index
df_train[feature_skewed] = np.log1p(df_train[feature_skewed])
print('%d features changed'%feature_skewed.shape[0])


# In[44]:


df_test[feature_skewed] = np.log1p(df_test[feature_skewed])
print('%d features changed'%feature_skewed.shape[0])


# ## 生成新特征

# In[45]:


corrmat = df_train.corr()
k = 20
cols = np.abs(corrmat).nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(25,25))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[46]:


cols = cols.drop('SalePrice')
for col in cols:
    df_train[col+'sqrt'] = np.sqrt(df_train[col])
    df_train[col+'2'] = df_train[col]**2
    df_train[col+'3'] = df_train[col]**3
    df_test[col+'sqrt'] = np.sqrt(df_test[col])
    df_test[col+'2'] = df_test[col]**2
    df_test[col+'3'] = df_test[col]**3

df_train.head(10)


# In[47]:


print(df_train.shape)
print('```````')
print(df_test.shape)


# ## 构造输入至模型中的训练集测试集。

# In[51]:


features = df_train.columns.drop(['SalePrice','Id'])
X_train = df_train[features]
y_train = df_train['SalePrice']
X_test = df_test[features]


# In[52]:


miss_data = pd.DataFrame({'amount': X_train.isnull().sum(), 'ratio':X_train.isnull().sum() / X_train.shape[0]})
print(miss_data)


# In[53]:


miss_data = pd.DataFrame({'amount': X_test.isnull().sum(), 'ratio':X_test.isnull().sum() / X_test.shape[0]})
print(miss_data)


# ## 构建模型

# ### 尝试使用线性进行拟合

# In[54]:


X = X_train.values
y = y_train.values
X_t = X_test.values

print(X.shape)
print(X_t.shape)


# In[48]:


from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import (RandomForestRegressor, 
                              AdaBoostRegressor, 
                              GradientBoostingRegressor, 
                              ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error


# In[ ]:


import xgboost as xgb


# In[55]:


ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
ridge.fit(X_train, y)
alpha = ridge.alpha_
print(alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                cv = 10)
ridge.fit(X_train, y)
alpha = ridge.alpha_
print("Best alpha :", alpha)


# In[54]:


ridge_train_out = ridge.predict(X)
ridge_test_out = ridge.predict(X_t)
print("RMSE of xgb regressor %f"
       %np.sqrt(mean_squared_error(y_true=y, y_pred=ridge_train_out)))


# In[55]:


result = pd.DataFrame()
result['Id'] = df_test['Id']
result['SalePrice'] = np.exp(ridge_test_out)


# In[56]:


result.to_csv('ridge_result.csv',index=False)


# In[57]:


lass = LassoCV(alphas = [0.0001, 0.0003, 0.001, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
lass.fit(X_train, y)
alpha = lass.alpha_
print(alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
lass = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                cv = 10)
lass.fit(X_train, y)
alpha = lass.alpha_
print("Best alpha :", alpha)


# In[58]:


lass_train_out = lass.predict(X)
lass_test_out = lass.predict(X_t)
print("RMSE of xgb regressor %f"
       %np.sqrt(mean_squared_error(y_true=y, y_pred=lass_train_out)))


# In[59]:


result = pd.DataFrame()
result['Id'] = df_test['Id']
result['SalePrice'] = np.exp(lass_test_out)
result.to_csv('lass_result.csv',index=False)


# In[60]:


en = ElasticNetCV(alphas = [0.0001, 0.0003, 0.001, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
en.fit(X_train, y)
alpha = en.alpha_
print(alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
en = ElasticNetCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                cv = 10)
en.fit(X_train, y)
alpha = en.alpha_
print("Best alpha :", alpha)


# In[61]:


en_train_out = en.predict(X)
en_test_out = en.predict(X_t)
print("RMSE of xgb regressor %f"
       %np.sqrt(mean_squared_error(y_true=y, y_pred=en_train_out)))


# In[62]:


result = pd.DataFrame()
result['Id'] = df_test['Id']
result['SalePrice'] = np.exp(en_test_out)
result.to_csv('en_result.csv',index=False)


# In[69]:


err_lowest = np.inf
estm_list = [1000, 2000, 500]
lr_list = [0.01, 0.05, 0.075]
for lr in lr_list:
    for estm in estm_list:
        params = {
            'learning_rate':lr,
            'n_estimators':estm,
            'max_depth':6,
            'min_child_weight':2,
            #gamma=1,
            'gamma':0.5,                        
            'subsample':0.4,
            'colsample_bytree':0.8,
            'objective': 'reg:linear',
            'nthread':-1,
            'scale_pos_weight':1
            }
    
        xgbm = SklearnRegressor(clf=xgb.XGBRegressor, params=params)
        err = cv_train(xgbm, X, y)
    
        if err < err_lowest:
            best_xgbm = xgbm
            err_lowest = err
            est_lowest = estm
            lr_lowest = lr

print(err_lowest)
print(est_lowest)
print(lr_lowest)


# In[70]:


xgb_train_out = best_xgbm.predict(X)
xgb_test_out = best_xgbm.predict(X_t)
print("RMSE of xgb regressor %f"
       %np.sqrt(mean_squared_error(y_true=y, y_pred=xgb_train_out)))

我可以在三个线性模型的基础上使用ensemble方法。
# In[72]:


print(ridge_train_out.shape)
print(lass_train_out.shape)
print(en_train_out.shape)
print(xgb_train_out.shape)


# In[76]:


ens_input = pd.DataFrame({'ridge':ridge_train_out.ravel(),
                          'lass':lass_train_out.ravel(),
                          'en':en_train_out.ravel(),
                          'xgb':xgb_train_out.ravel(),
#                           'svr':svr_oof_train.ravel()
                         })

ens_test = pd.DataFrame({'ridge':ridge_test_out.ravel(),
                          'lass':lass_test_out.ravel(),
                          'en':en_test_out.ravel(),
                          'xgb':xgb_test_out.ravel(),
#                           'svr':svr_oof_test.ravel()
                        })

ens_input.head(5)


# In[78]:


ens_EN = ElasticNetCV(alphas = [0.0001, 0.0003, 0.001, 0.003])
ens_EN.fit(ens_input, y)
alpha = ens_EN.alpha_
print(alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
ens_EN = ElasticNetCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                cv = 10)
ens_EN.fit(ens_input, y)
alpha = ens_EN.alpha_
print("Best alpha :", alpha)


# In[79]:


ens_train_out = ens_ridge.predict(ens_input)
ens_test_out = ens_ridge.predict(ens_test)
print("RMSE of xgb regressor %f"
       %np.sqrt(mean_squared_error(y_true=y, y_pred=ens_train_out)))


# In[80]:


result = pd.DataFrame()
result['Id'] = df_test['Id']
result['SalePrice'] = np.exp(ens_test_out)
result.to_csv('ens_result.csv',index=False)


# # 建立机器学习模型，进行训练

# In[64]:


import xgboost as xgb
from sklearn.ensemble import (RandomForestRegressor, 
                              AdaBoostRegressor, 
                              GradientBoostingRegressor, 
                              ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

这里先分析一下整体的思路。
整个过程分为两步。首先，训练第一层的模型，这里使用sklearn库中的模型；第二部将第一层模型的输出作为输入feeddaoxgboost进行回归。最终的回归结果为输出。先说一下第一步的总体思路。
为了封装，将第一步中使用的模型使用面向对象编程方法，统一接口。
接口函数包括init，train，predict
训练使用out-of-fold方法，即对训练集进行划分，并采用交叉验证集进行验证。
# ## 第一步，是建立具有统一接口的类

# In[65]:


class SklearnRegressor(object):
    def __init__(self, clf,params=None):
        self.clf = clf(**params)
        
    def train(self, X, y):
        self.clf.fit(X, y)
    
    def predict(self, X):
        return self.clf.predict(X)
    
    def fit(self, X, y ):
        return self.clf.fit(X, y)


# In[66]:


ntrain = X_train.shape[0]
ntest = X_test.shape[0]
SEED = 0
NFOLDS = 5
kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED)

在建立接口，并准备好数据后，就是机器学习模型部分。
第一层模型的训练，具体的步骤如下：
1，选择模型
2，预设参数
3，模型训练并交叉验证第一次迭代中，我使用预设的参数对模型进行设置。
在之后的迭代中，需要使用交叉验证法，寻找最优参数。
# ## 应用OOF方法，训练第一层模型

# In[67]:


def oof_train(clf, X, y, X_test):
# def oof_train(clf, X, y):
    train_out = np.zeros((ntrain,))
    test_out = np.zeros((ntest,))
    oof_out = np.zeros((NFOLDS, ntest))
    
    for i, (train_index, val_index) in enumerate(kf):
        x_tr = X[train_index]
        y_tr = y[train_index]
        x_val = X[val_index]
        
        clf.train(x_tr, y_tr)
        
        train_out[val_index] = clf.predict(x_val)
        oof_out[i, :] = clf.predict(X_test)
    
    test_out = oof_out.mean(axis=0)
    return train_out.reshape(-1, 1), test_out.reshape(-1, 1)


# ## 初步设定模型参数

# In[ ]:


# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 5000,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0,
    'random_state' : SEED
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':5000,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0,
    'random_state' : SEED
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 2000,
    'learning_rate' : 0.75,
    'random_state' : SEED
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 2000,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0,
    'random_state' : SEED
}

# Support Vector Regressor parameters 
svr_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }


# 按照预设参数，初始化模型

# In[ ]:


rf = SklearnRegressor(clf=RandomForestRegressor, params=rf_params)
et = SklearnRegressor(clf=ExtraTreesRegressor, params=et_params)
ada = SklearnRegressor(clf=AdaBoostRegressor, params=ada_params)
gb = SklearnRegressor(clf=GradientBoostingRegressor, params=gb_params)
# svr = SklearnRegressor(clf=SVR, params=svr_params)


# 在这里，插入对于cross validation的使用。
# 使用参数较少的svr进行试验。

# In[ ]:


X = X_train.values
y = y_train.values
X_t = X_test.values

print(X.shape)
print(X_t.shape)


# In[68]:


def cv_train(clf, X, y):
    train_out = np.zeros((ntrain,))
    
    for i, (train_index, val_index) in enumerate(kf):
        x_tr = X[train_index]
        y_tr = y[train_index]
        x_val = X[val_index]
        
        clf.train(x_tr, y_tr)
        train_out[val_index] = clf.predict(x_val)
   
    error = np.sqrt(mean_squared_error(y_true=y, y_pred=train_out))
    return error


# In[ ]:


c_list = [0.005, 0.01, 0.015, 0.02]
lowest_err = np.inf
for c in c_list:
    params = {
        'kernel': 'linear',
        'C': c
    }
    svr = SklearnRegressor(clf=SVR, params=params)
    err = cv_train(svr, X, y)
    
    if err < lowest_err:
        best_svr = svr
        lowest_err = err
        lowest_c = c

print(lowest_err)
print(lowest_c)


# In[ ]:


# Random Forest parameters
# rf_params = {
#     'n_jobs': -1,
#     'n_estimators': 500,
#      'warm_start': True, 
#      #'max_features': 0.2,
#     'max_depth': 6,
#     'min_samples_leaf': 2,
#     'max_features' : 'sqrt',
#     'verbose': 0,
#     'random_state' : SEED
# }

estimators_list = [10, 50, 100, 150,200]
lowest_err = np.inf
for est in estimators_list:
    # Random Forest parameters
    params = {
        'n_jobs': -1,
        'n_estimators': est,
        'warm_start': True, 
        #'max_features': 0.2,
        'max_depth': 6,
        'min_samples_leaf': 2,
        'max_features' : 'sqrt',
        'verbose': 0,
        'random_state' : SEED
}
    rf = SklearnRegressor(clf=RandomForestRegressor, params=params)
    err = cv_train(rf, X, y)
    
    if err < lowest_err:
        best_rf = rf
        lowest_err = err
        lowest_est = est

print(lowest_err)
print(lowest_est)


# In[ ]:


# Extra Trees Parameters
# et_params = {
#     'n_jobs': -1,
#     'n_estimators':500,
#     #'max_features': 0.5,
#     'max_depth': 8,
#     'min_samples_leaf': 2,
#     'verbose': 0,
#     'random_state' : SEED
# }

estimators_list = [170,200,250]
lowest_err = np.inf
for est in estimators_list:
    # Random Forest parameters
# Extra Trees Parameters
    params = {
    'n_jobs': -1,
    'n_estimators':est,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0,
    'random_state' : SEED
}
    et = SklearnRegressor(clf=ExtraTreesRegressor, params=params)
    err = cv_train(et, X, y)
    
    if err < lowest_err:
        best_et = et
        lowest_err = err
        lowest_est = est

print(lowest_err)
print(lowest_est)


# In[ ]:


# AdaBoost parameters
# ada_params = {
#     'n_estimators': 500,
#     'learning_rate' : 0.75,
#     'random_state' : SEED
# }

lr_list = [0.08, 0.1, 0.15]
est_list = [80, 100, 150, 200]
lowest_err = np.inf
for lr in lr_list:
    for est in est_list:
        params = {
         'n_estimators': est,
         'learning_rate' : lr,
         'random_state' : SEED
        }
        
        ada = SklearnRegressor(clf=AdaBoostRegressor, params=params)
        err = cv_train(ada, X, y)
        
        if err < lowest_err:
            best_ada = ada
            lowest_err = err
            lowest_est = est
            lowest_lr = lr

print(lowest_err)
print(lowest_est)
print(lowest_lr)


# In[ ]:


est_list = [10,30, 50,70]
lowest_err = np.inf
for est in est_list:
    params = {
    'n_estimators':est,
    #'max_features': 0.5,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0,
    'random_state' : SEED
}
    gb = SklearnRegressor(clf=GradientBoostingRegressor, params=params)
    err = cv_train(gb, X, y)
    
    if err < lowest_err:
        best_gb = gb
        lowest_err = err
        lowest_est = est

print(lowest_err)
print(lowest_est)


# ## 将输入数据转化为ndarray类型

# In[ ]:


# et_oof_train, et_oof_test = oof_train(best_et, X, y, X_t)
# rf_oof_train, rf_oof_test = oof_train(best_rf, X, y, X_t)
# ada_oof_train, ada_oof_test = oof_train(best_ada, X, y, X_t)
# gb_oof_train, gb_oof_test = oof_train(best_gb, X, y, X_t)
# svr_oof_train, svr_oof_test = oof_train(best_svr, X, y, X_t)

et_oof_train, et_oof_test = oof_train(et, X, y, X_t)
rf_oof_train, rf_oof_test = oof_train(rf, X, y, X_t)
ada_oof_train, ada_oof_test = oof_train(ada, X, y, X_t)
gb_oof_train, gb_oof_test = oof_train(gb, X, y, X_t)
# svr_oof_train, svr_oof_test = oof_train(svr, X, y, X_t)
print('Training is completed!')

到这里，初步完成了对于第一层分类器的训练，下面看一下分类器的在训练集上的正确率。
# In[ ]:


print("RMSE of et regressor %f"
       %np.sqrt(mean_squared_error(y_true=y, y_pred=et_oof_train)))
print("RMSE of rf regressor %f"
       %np.sqrt(mean_squared_error(y_true=y, y_pred=rf_oof_train)))
print("RMSE of ada regressor %f"
       %np.sqrt(mean_squared_error(y_true=y, y_pred=ada_oof_train)))
print("RMSE of gb regressor %f"
       %np.sqrt(mean_squared_error(y_true=y, y_pred=gb_oof_train)))
# print("RMSE of svr regressor %f"
#        %np.sqrt(mean_squared_error(y_true=y, y_pred=svr_oof_train)))

当前，我初步得到了第一层4个回归期的输出，下面要将这些输出作为输入，训练第二层的分类器xgboost。
但在具体的工作之前，还需要对输出的四个参数进行一定的相关性分析。
# In[ ]:


xgb_input = pd.DataFrame({'rf':rf_oof_train.ravel(),
                          'et':et_oof_train.ravel(),
                          'ada':ada_oof_train.ravel(),
                          'gb':gb_oof_train.ravel(),
#                           'svr':svr_oof_train.ravel()
                         })

xgb_test = pd.DataFrame({'rf':rf_oof_test.ravel(),
                          'et':et_oof_test.ravel(),
                          'ada':ada_oof_test.ravel(),
                          'gb':gb_oof_test.ravel(),
#                           'svr':svr_oof_test.ravel()
                        })

xgb_input.head(5)

原则上，希望四个初级分类器的输出之间的互相关性很小，这样每个分类器都能够提供独特的信息。
# In[ ]:


corrmatrix = xgb_input.corr()
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmatrix, square=True)


# In[ ]:


xgb_X = xgb_input.values
xgb_y = y_train.ravel()

xgb_test_X = xgb_test.values


# In[ ]:


X = X_train.values
y = y_train.values
X_t = X_test.values

print(X.shape)
print(X_t.shape)

现在已经得到需要输入到最后xgb回归器中的数据，下面就是要调用xgb的库函数，对数据进行oof训练了。
# In[ ]:


xgbm = xgb.XGBRegressor(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.5,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'reg:linear',
 nthread= -1,
 scale_pos_weight=1)

xgb_train_out = np.zeros((ntrain,))
xgb_test_out = np.zeros((ntest,))

for i, (train_index,val_index) in enumerate(kf):
    Xf = X[train_index]
    yf = y[train_index]
    xgbm.fit(X=Xf, y=yf)
    xgb_train_out[val_index] = xgbm.predict(X[val_index])
    
xgb_test_out = xgbm.predict(X_t)


# In[ ]:


err_lowest = np.inf
estm_list = [100, 200, 300]
lr_list = [0.01, 0.05, 0.075]
for lr in lr_list:
    for est in estm_list:
        params = {
            'learning_rate':lr,
            'n_estimators':est,
            'max_depth':6,
            'min_child_weight':4,
            #gamma=1,
            'gamma':0.5,                        
            'subsample':0.6,
            'colsample_bytree':0.8,
            'objective': 'reg:linear',
            'nthread':-1,
            'scale_pos_weight':1
            }
    
        xgbm = SklearnRegressor(clf=xgb.XGBRegressor, params=params)
        err = cv_train(xgbm, X, y)
    
        if err < err_lowest:
            best_xgbm = xgbm
            err_lowest = err
            est_lowest = est
            lr_lowest = lr

print(err_lowest)
print(est_lowest)
print(lr_lowest)


# In[ ]:


xgb_train_out = best_xgbm.predict(X)
xgb_test_out = best_xgbm.predict(X_t)

现在得到了第二层回归器xgboost在训练集上的输出，验证一下现在相对于之前的一级回归器是否有提升
# In[ ]:


print("RMSE of xgb regressor %f"
       %np.sqrt(mean_squared_error(y_true=y, y_pred=xgb_train_out)))


# In[ ]:


result = pd.DataFrame()
result['Id'] = df_test['Id']
result['SalePrice'] = np.exp(xgb_test_out)


# In[ ]:


result.to_csv('result.csv',index=False)


# # 总结
# 可以看出，第二级的回归器在第一级的基础上有一定的提升，但提升有限。 这说明： 1，ensemble的思路具有可行性，因为确实得到了提升； 2，第一级回归器及第二级回归器在当前并没有使用交叉验证调参的方法，这是下一步需要引入的内容； 3，四个初级回归器之间相关性较高，探索降低相关性或其他方法； 4，引入更加强大的回归器用于第一二甚至三级回归器。

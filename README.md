# 金融业数字化转型技能大赛建模（银行组）——信贷风险预测模型

## 1. 项目简介
本项目基于贷款基础表和银行交易流水表，通过多维度特征工程和机器学习方法，构建用于信贷风险预测的模型。赛题的解决思路如下：  

在数据预处理和特征工程阶段有两套方案（以下称为预处理特征工程一、预处理特征工程二），分别对训练集和预测集进行处理，尽可能丰富了模型的特征。在模型训练阶段，采用AutoGluon框架，对预处理特征工程一采用不同的参数训练得到模型一和模型二，对预处理特征工程二的数据训练得到模型三。最后分别对三个模型预测的结果加权后得到最终预测。

具体步骤如下：
- 读取原始数据，数据预处理
- 对借贷基础数据进行特征工程（如：周期性时间编码、比例特征、交互项）
- 对流水数据进行多层次聚合与统计（如：时间窗口内的流入/流出资金）
- 衍生交易频率、时间规律性、时间衰减特征和异常检测特征等
- 融合所有特征，输入AutoGluon自动建模与调参
- 模型预测，概率加权

核心优势为：  
- 综合客户基本信息、交易流水及行为模式，提取高维度特征；  
- 引入交易时间衰减、行为异常检测和交易序列等高级特征；  
- 使用 AutoGluon 提供的自动化模型训练和集成方法，实现高效预测。
- 分别训练得到三个预测模型，综合模型表现，加权预测结果得到最终预测。

---

## 2. 数据预处理

### 2.1 数据预处理一

#### 时间戳处理
- 转换 `issue_time`, `record_time`, `history_time` 等时间戳为datetime格式
- 统一处理训练集和测试集的时间格式

#### 类别特征编码
- **重要类别特征** (residence, title, career, zip_code): 转换为字符串 + 目标编码
- **其他类别特征**: 使用LabelEncoder进行数值编码  

#### 数据清洗
- 删除常数列 (nunique <= 1)
- 处理无穷值 (用列的最大/最小值替换)
- 删除40个预先定义的无效特征

### 2.2 数据预处理二

#### 时间戳处理
将时间戳转换为日期时间格式，并提取年、月、日、星期、是否周末、是否月初、是否月末、小时等时间特征。

#### 交易标签编码
区分流入和流出交易（direction=0为支出，direction=1为收入）。

---

## 3. 特征工程

### 3.1 特征工程一

#### 3.1.1基础特征工程

##### 时间特征
- **基础时间特征**: 年、月、季度、星期几
- **周期性编码**: sin/cos编码月份、星期、季度
- **时间间隔**: 账户年龄、信用历史天数、贷款年龄

##### 数值特征衍生
- **比率特征**: 贷款期限比、分期付款比率、贷款限额比等
- **多项式特征**: 平方、平方根、对数变换
- **交互特征**: 数值特征间的乘法和除法交互

##### 风险代理特征
- 高利率标识、长期贷款标识、高余额标识、联合贷款标识

#### 3.1.2 银行流水特征

##### 基础聚合特征
- 入金/出金统计量 (总和、均值、标准差、最大值、计数)
- 净资金流、入金比例、出金比例
- 对数变换关键金额特征

##### 时间窗口特征
- **贷前窗口**: [7, 15, 30, 60, 90]天的交易统计
- **贷后窗口**: [7, 15, 30, 60, 90]天的交易统计
- 资金流变化率趋势分析

##### 行为模式特征
- **交易规律性**: 交易间隔统计、时间规律性
- **金额模式**: 整百/整千交易、小额出金统计
- **时间段偏好**: 夜间、白天、傍晚交易统计
- **交易集聚度**: 赫芬达尔指数、最大日交易量

#### 3.1.3 高级特征工程

##### 时间衰减加权特征
- 基于指数衰减函数的近期行为加权
- 加权资金流、近期活动强度、交易频率

##### 异常检测特征
- 金额偏度/峰度、异常值比例
- 时间聚类得分、行为波动性
- 紧急交易检测、非工作时间交易比例

##### 交易序列特征
- 将交易序列转换为文本格式 ("in_金额分桶 out_金额分桶")
- TF-IDF序列向量化 + SVD降维

#### 3.1.4 业务特征工程

##### 偿债能力特征
- 债务覆盖比率、可支配收入
- 流动性比率、可支撑天数

##### 趋势特征
- 收入增长趋势、资金流变化趋势
- 波动性特征 (标准差、变异系数)

#### 3.1.5 特征优化

##### 特征增强
- 重要特征的log、sqrt、平方变换
- 分箱特征创建
- 深度交互特征构建

##### 特征选择
- 基于重要性分析删除低价值特征
- 保留与业务逻辑高度相关的特征

##### 目标编码
- 对重要类别特征进行平滑目标编码
- 保持训练集和测试集编码的一致性

### 3.2 特征工程二

#### 3.2.1 银行流水特征工程

- 按用户ID分组，计算以下特征：
- **交易活跃度特征**
- 总交易次数、流入流出次数、流入流出比例、活跃天数、月度活跃度等。
- **交易金额特征**
- 总流入流出金额、净流入流出、金额统计特征（均值、标准差、最大值、最小值、中位数、变异系数等）、金额分布特征（偏度、峰度）。
- **交易时间规律特征**
- 周末交易比例、月初月末交易比例、工作日与周末交易比例、夜间交易比例、工作时间交易比例等。
- **大额交易特征**
- 设定大额阈值（5000）和小额阈值（100），统计大额和小额交易的次数和比例。
- **交易间隔特征**
- 平均交易间隔、最小、最大、标准差、中位数、变异系数。
- **交易小时分布**
- 平均交易小时、标准差、最小、最大。
- **交易金额变化趋势**
- 使用线性拟合得到流入、流出和净流量的趋势斜率。
- **金额比率特征**
- 平均流入流出金额比率、中位数比率、最大金额比率等。

#### 3.2.2 主表特征工程

- **时间特征处理**
- 将issue_time, record_time, history_time转换为日期时间，并计算时间间隔特征。
- **邮政编码特征**
- 提取前三位作为地区特征。
- **贷款特征工程**
- 计算每期贷款金额、分期付款比率、月付款、债务收入比等。
- **账户特征**
- 利用率、活跃账户比例、非活跃账户数、平均账户余额、平均账户限额等。
- **信用等级编码**
- 将等级映射为数字，并创建是否为高等级的虚拟变量。
- **时间相关特征**
- 发放贷款的月份、季度、星期几、是否周末等。
- **数值特征的多项式和交互特征**
- 贷款金额的平方、平方根、对数，以及贷款期限和金额的交互项等。

---

## 4. 模型算法简要介绍

AutoGluon 是亚马逊开发的 AutoML 框架，核心功能：

- 自动识别数据类型；
- 自动特征工程（编码、填补、变换）；
- 自动模型选择与融合（Stacking / Ensembling）；
- 自动调参与验证。

内部算法集成：AutoGluon 集成多种模型（并自动融合）

- 树模型类：LightGBM、CatBoost、XGBoost
- 线性模型类：Logistic Regression
- 神经网络类：TabTransformer / MLP
- Bagging + Stacking：自动加权融合各模型结果

---

## 5 模型实现步骤

我们已经将代码封装成了train.py和test.py文件，分别用来训练模型和预测结果，可以直接调用.py文件完成训练任务和预测任务。其中，train.py和test.py文件的数据预处理和特征工程部分的函数一致，区别为train.py文件包含了模型训练部分，并将训练好的模型保存在~/code/model下；而test.py文件在数据预处理和特征工程后直接调用~/code/model下已经训练好的模型文件完成预测，并将预测结果保存至～/result下。具体步骤如下：

### 数据预处理与特征工程

如上所述，我们分别进行两套数据预处理和特征工程，基于银行流水和主表进行特征提取，然后合并所有特征。

### 数据准备

将训练集和测试集的特征合并，并确保id和label列正确处理。

### 模型构建、模型训练

由于Autogluon框架可以实现自动模型选择与融合、自动调参与验证，因此模型构建和模型训练可以一步完成。三个模型的模型构建和模型训练如下：

```
#### 模型一
predictor_one = TabularPredictor(
    label='label',
    problem_type='binary',
    eval_metric='roc_auc',
    #path='AutogluonModels/model1'
).fit(
    train_data=df_train[cols_input+['label']],
    tuning_data=None,
    presets='best_quality',
    time_limit=7200,
    hyperparameters={
    'GBM': {
        'reg_lambda': 1.0,
        'reg_alpha': 0.5,
        'feature_fraction': 0.75,
        'num_leaves': 48,
        'min_child_samples': 30,
    },
    'XGB': {
        'lambda': 1.0,
        'alpha': 0.5,
        'max_depth': 7,
        'colsample_bytree': 0.75,
        'subsample': 0.85,
    },
    'CAT': {
        'l2_leaf_reg': 2,
        'depth': 7,
        'random_strength': 1.0,
        'bagging_temperature': 0.9,
    },
    'RF': {
        'max_features': 0.7,
        'max_depth': 18,
        'n_estimators': 150,
        'min_samples_leaf': 3,
    },
    'XT': {
        'max_features': 0.75,
        'max_depth': 15,
        'n_estimators': 120,
        'min_samples_leaf': 2,
    }
}
)

#### 模型二

predictor_two = TabularPredictor(
    label='label',
    problem_type='binary',
    eval_metric='roc_auc',
    #path='AutogluonModels/model2'
).fit(
    train_data=df_train[cols_input+['label']],
    tuning_data=None,
    presets='best_quality',
    time_limit=7200,
    hyperparameters={
    'GBM': {'reg_lambda': 2.0, 'reg_alpha': 1.0},
    'XGB': {'lambda': 2.0, 'alpha': 1.0},
    'CAT': {'l2_leaf_reg': 3},
    'RF': {},
    'XT': {}
    }
)

#### 模型三

predictor_three = TabularPredictor(label="label", problem_type="binary", eval_metric="roc_auc").fit(
    train_data=df_train1[cols_input+['label']],
    presets="best_quality",  # 可以选 fast_training / best_quality
    hyperparameters={
        "GBM": {},      # LightGBM
        "CAT": {},      # CatBoost
        "XGB": {},      # XGBoost
        "RF": {},       # 随机森林
        "XT": {},       # ExtraTrees
        # "NN_TORCH": {}   # 默认包含神经网络，如果不要就直接去掉
    }
)
```

### 模型评估

直接调用AutoGluon框架的leaderboard评估模型，得到最优的模型

```
lb_1 = predictor_one.leaderboard(silent=True)
lb_2 = predictor_two.leaderboard(silent=True)
lb_3 = predictor_three.leaderboard(silent=True)
```

### 模型预测

从model加载三个训练好的模型后，调用模型完成预测，然后加权得到最终预测，将文件保存至/result下

```
#模型预测一

y_pred_proba_1 = predictor_one.predict_proba(df_testA[cols_input+['label']])[1]
output_df_1 = pd.DataFrame({'id': df_testA['id'], 'label': y_pred_proba_1})

#模型预测二
y_pred_proba_2 = predictor_two.predict_proba(df_testA[cols_input+['label']])[1]
output_df_2 = pd.DataFrame({'id': df_testA['id'], 'label': y_pred_proba_2})

#模型预测三
y_pred_proba_3 = predictor_three.predict_proba(df_testA1[cols_input+['label']])[1
output_df_3 = pd.DataFrame({'id': df_testA1['id'], 'label': y_pred_proba_3})

output_df = output_df_1[['id']].copy()
output_df['label'] = output_df_1['label'] *0.75 + output_df_2['label'] * 0.2 + output_df_3['label'] *0.05

output_df.to_csv('/数智先锋CCB_刘德华/result/result.csv', index=False, header=True)
```

---

## 6 算法运行环境

代码运行在Google Colab环境中，需要安装AutoGluon库。具体步骤包括：
- 挂载Google Drive以便访问数据文件。
- 安装AutoGluon：!pip install autogluon
- 导入必要的库，包括pandas、numpy、scipy等。
- 执行特征工程和模型训练。
必要的包见requirements.txt




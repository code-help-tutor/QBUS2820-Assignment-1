# QBUS2820 Assignment 1 (30 marks)

**Date**: August 23, 2024

## 1. Background
Developing a predictive model for building heating load is crucial for energy efficiency management. You are working for an energy efficiency consulting firm and tasked with optimizing building heating system operations by predicting daily heating load requirements.

The `HeatingLoad` variable in the dataset `HeatingLoad_training.csv` represents the daily energy needed to maintain comfortable indoor temperatures. The data includes predictors such as building characteristics, environmental conditions, and occupancy. The response variable and covariates are detailed in Table 1.

**Table 1: Description of Variables**
| Variable | Description |
| --- | --- |
| HeatingLoad | Total daily heating energy required (in kWh) |
| BuildingAge | Age of the building (in years) |
| BuildingHeight | Height of the building (in meters) |
| Insulation | Insulation quality (1 = Good, 0 = Poor) |
| AverageTemperature | Average daily temperature (in °C) |
| SunlightExposure | Solar energy received per unit area (in W/m²) |
| WindSpeed | Wind speed at the building’s location (in m/s) |
| OccupancyRate | Proportion of the building that is occupied (percentage) |

Your task is to develop a regression model to predict `HeatingLoad` based on these covariates. Additionally, you are provided with the dataset `HeatingLoad_test_without_HL.csv`, which is the real test dataset `HeatingLoad_test.csv` with the `HeatingLoad` column removed. The test dataset has the same structure as the training data.

### 1.1 Test Error
To measure prediction accuracy, use mean squared error (MSE) on the test data. Let $\hat{y}_i$ be the prediction of $y_i$, where $y_i$ is the i-th `HeatingLoad` in the test data. The test error is computed as follows:

Test error = $\frac{1}{n_{test}}\sum_{y_i\in\text{test data}}(\hat{y}_i - y_i)^2$, where $n_{test}$ is the number of observations in the test data.

## 2. Submission Instructions
1. Submit THREE files (or more if necessary) via the Canvas site:
    - A document file named `SID_Assignment1_document.pdf`, reporting your data analysis procedure and results. Replace “SID” with your student ID.
    - A Python file named `SID_Assignment1_implementation.ipynb` that implements your data analysis procedure and produces the test error. Submit additional files if needed following the format `SID_Assignment1`.
    - A CSV file `SID_Assignment1_HL_prediction.csv` containing the predictions of `HeatingLoad` for the dataset `HeatingLoad_test_without_HL.csv`. This CSV file should have only one column named `HeatingLoad` holding the predicted values.
2. Regarding your document file `SID_Assignment1_document.pdf`:
    - Detail your data analysis procedure: how Exploratory Data Analysis (EDA) was conducted, the methods/predictors used, and the reasoning behind them. Report all numerical results to four decimal places.
    - Present relevant graphs and tables clearly and appropriately.
    - Page limit is 15 pages including appendices, computer output, graphs, tables, etc.
3. The Python file must be written using Jupyter Notebook assuming all necessary data files (`HeatingLoad_training.csv` and `HeatingLoad_test.csv`) are in the same folder as the Python file.
    - The Python file `SID_Assignment1_implementation.ipynb` must include the following code in the last code cell:
    ```python
    import pandas as pd
    HeatingLoad_test = pd.read_csv("HeatingLoad_test.csv")
    # YOUR CODE HERE: code that produces the test error test_error
    print(test_error)
    ```
    - The marker expects to see the same test error as if provided with the complete test data. The file should contain enough explanations for the marker to run your code.
    - Use only methods covered in lectures and tutorials. You can use any publicly available Python libraries to implement your models.

## 3. Marking Criteria
This assignment is worth 30 marks total, with 18 marks for the content of `SID_Assignment1_document.pdf` and 12 marks for the Python implementation.

1. Prediction accuracy:
    - The marker first runs `SID_Assignment1_implementation.ipynb`.
    - If the file runs smoothly and produces a test error, up to 12 marks will be awarded based on prediction accuracy relative to the smallest MSE and the appropriateness of your implementation.
    - If the marker cannot run the file or if no test error is produced, partial marks (maximum 4) may be awarded based on the appropriateness of the file.
2. Report in `SID_Assignment1_document.pdf`: Up to 18 marks are allocated based on:
    - Appropriateness of the chosen prediction method.
    - Detail, discussion, and explanation of your data analysis procedure.
3. CSV File Submission: Up to 2 marks will be deducted if you fail to upload the CSV file in the correct format.

## 4. Errors
If you believe there are errors in this assignment, please contact the teaching team.


# QBUS2820 Assignment 1

## 一、任务概述
1. **背景**：开发建筑热负荷预测模型对于能源效率管理至关重要。作为能源效率咨询公司的员工，任务是通过预测建筑物的每日热负荷需求来优化建筑物的供暖系统运行。
2. **数据集**：提供了`HeatingLoad_training.csv`数据集，其中`HeatingLoad`表示维持建筑物内舒适温度所需的每日能量（以 kWh 为单位），还有其他影响热负荷的预测变量，如建筑年龄、建筑高度、保温质量、平均温度、日照暴露、风速和占用率等。另外还有`HeatingLoad_test_without_HL.csv`数据集，即真实测试数据集`HeatingLoad_test.csv`去掉了`HeatingLoad`列。
3. **任务要求**：
   - 开发一个回归模型来基于这些协变量预测`HeatingLoad`。
   - 使用均方误差（MSE）在测试数据上衡量预测准确性。
   - 通过 Canvas 提交三个文件：
     - 文档文件`SID_Assignment1_document.pdf`，报告数据分析过程和结果。
     - Python 文件`SID_Assignment1_implementation.ipynb`，实现数据分析过程并产生测试误差。
     - CSV 文件`SID_Assignment1_HL_prediction.csv`，包含对`HeatingLoad_test_without_HL.csv`数据集的`HeatingLoad`预测值。

## 二、数据分析过程

### （一）探索性数据分析（EDA）
1. **数据读取与初步观察**：
   - 使用`pandas`库读取`HeatingLoad_training.csv`数据集，查看数据集的大小、列名、数据类型等基本信息。
   - 输出数据集的前几行，了解数据的大致分布情况。
2. **变量分布分析**：
   - 对于数值型变量（如`BuildingAge`、`BuildingHeight`、`AverageTemperature`、`SunlightExposure`、`WindSpeed`、`OccupancyRate`），绘制直方图和箱线图，观察其分布形态、异常值情况。
   - 对于分类变量（如`Insulation`），绘制柱状图，查看不同类别所占比例。
3. **相关性分析**：
   - 计算所有变量之间的相关性系数矩阵，使用热图可视化相关性。重点关注`HeatingLoad`与其他变量之间的相关性。

### （二）方法选择与预测变量确定
1. **选择回归模型**：
   - 考虑到任务是预测连续型变量，选择线性回归模型作为基础模型。线性回归模型具有简单直观、易于解释的优点，并且在很多实际问题中都有较好的表现。
   - 同时，也可以考虑使用更复杂的回归模型，如岭回归、Lasso 回归等，以防止过拟合。
2. **预测变量选择**：
   - 根据相关性分析的结果，选择与`HeatingLoad`相关性较高的变量作为预测变量。例如，如果`AverageTemperature`和`Insulation`与`HeatingLoad`有较高的相关性，那么可以将这两个变量作为预测变量。
   - 也可以通过逐步回归等方法自动选择最优的预测变量组合。

### （三）模型训练与评估
1. **数据预处理**：
   - 对数值型变量进行标准化处理，使不同变量具有相同的尺度，避免某些变量对模型的影响过大。
   - 将分类变量进行独热编码，以便模型能够处理。
2. **模型训练**：
   - 使用训练数据集对选择的回归模型进行训练。
   - 可以使用交叉验证等方法来选择最优的模型参数。
3. **模型评估**：
   - 在训练集上评估模型的性能，计算训练集的均方误差（MSE）、决定系数（R²）等指标。
   - 使用测试数据集对模型进行最终评估，计算测试集的 MSE，与其他提交的结果进行比较，确定预测准确性。

## 三、Python 实现

以下是使用 Jupyter Notebook 实现数据分析过程的示例代码：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

# 读取训练数据集
data = pd.read_csv("HeatingLoad_training.csv")

# 探索性数据分析
# 变量分布分析
numerical_vars = ['BuildingAge', 'BuildingHeight', 'AverageTemperature', 'SunlightExposure', 'WindSpeed', 'OccupancyRate']
for var in numerical_vars:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[var], kde=True)
    plt.title(f'Distribution of {var}')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(data[var])
    plt.title(f'Boxplot of {var}')
    plt.show()

categorical_vars = ['Insulation']
for var in categorical_vars:
    plt.figure(figsize=(8, 6))
    sns.countplot(data[var])
    plt.title(f'Distribution of {var}')
    plt.show()

# 相关性分析
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 方法选择与预测变量确定
# 假设根据相关性分析选择 AverageTemperature、Insulation 和 OccupancyRate 作为预测变量
X = data[['AverageTemperature', 'Insulation', 'OccupancyRate']]
y = data['HeatingLoad']

# 数据预处理
scaler = StandardScaler()
X[numerical_vars] = scaler.fit_transform(X[numerical_vars])

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X[['Insulation']]).toarray()
X = np.hstack((X[numerical_vars].values, X_encoded))

# 模型训练
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Test MSE: {mse:.4f}')

# 对测试数据集进行预测
test_data = pd.read_csv("HeatingLoad_test_without_HL.csv")
X_test_new = test_data[['AverageTemperature', 'Insulation', 'OccupancyRate']]
X_test_new[numerical_vars] = scaler.transform(X_test_new[numerical_vars])
X_test_new_encoded = encoder.transform(X_test_new[['Insulation']]).toarray()
X_test_new = np.hstack((X_test_new[numerical_vars].values, X_test_new_encoded))

predictions = model.predict(X_test_new)

# 保存预测结果到 CSV 文件
pd.DataFrame({'HeatingLoad': predictions}).to_csv("SID_Assignment1_HL_prediction.csv", index=False)

import pandas as pd
HeatingLoad_test = pd.read_csv("HeatingLoad_test.csv")
# YOUR CODE HERE: code that produces the test error test_error
y_true = HeatingLoad_test['HeatingLoad']
y_pred_test = pd.read_csv("SID_Assignment1_HL_prediction.csv")['HeatingLoad']
test_error = mean_squared_error(y_true, y_pred_test)
print(test_error)
```

请注意，以上代码仅为示例，实际实现中可能需要根据具体情况进行调整和优化。

## 四、结果呈现
1. 在文档文件`SID_Assignment1_document.pdf`中，详细呈现数据分析过程，包括 EDA 的结果（相关图表）、方法选择的理由、模型训练和评估的结果等。
2. 在 Python 文件中，确保代码能够正确运行并输出测试误差。
3. 提交的 CSV 文件格式正确，包含对测试数据集的`HeatingLoad`预测值。

## 五、注意事项
1. 严格按照提交要求提交文件，确保文件名正确。
2. 文档文件的篇幅不超过 15 页，包括所有内容（附录、计算机输出、图表、表格等）。
3. Python 文件必须使用 Jupyter Notebook 编写，并假设所有必要的数据文件都与 Python 文件在同一文件夹中。
4. 仅使用课程和教程中涵盖的方法，可以使用任何公开可用的 Python 库来实现模型。
5. 如果未能以正确格式上传 CSV 文件，最多将扣除 2 分。


# QBUS2820 Assignment 1

# CS Tutor | 计算机编程辅导 | Code Help | Programming Help

# WeChat: cstutorcs

# Email: tutorcs@163.com

# QQ: 749389476

# 非中介, 直接联系程序员本人

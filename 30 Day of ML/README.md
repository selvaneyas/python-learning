# 📌 Day 1: Introduction to Machine Learning  
**Machine Learning (ML)** is a field of Artificial Intelligence that enables computers to learn from data without being explicitly programmed.  

## 🔹 What You’ll Learn Today:  
✅ What is Machine Learning?  
✅ Types of ML (Supervised, Unsupervised, Reinforcement Learning)  
✅ Real-world applications  
✅ Setting up Python environment  
✅ Writing your first ML script  

Run the following command to install the required libraries:  
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```



# 📌 Day 2: Essential Python Libraries for ML  

Today, we explored the key Python libraries for Machine Learning:  

## 🔹 Libraries Covered  
✅ **NumPy** – Numerical computations  
✅ **Pandas** – Data manipulation  
✅ **Matplotlib & Seaborn** – Data visualization  
✅ **Scikit-Learn** – ML model training  




# 📌 Day 3: Data Preprocessing & Cleaning in ML  

## 🔹 Why is Data Preprocessing Important?  
Raw data can be messy, so we need to clean and transform it before using it in ML models.  

## 🔹 Key Steps Covered  
✅ Handling Missing Data  
✅ Removing Duplicates  
✅ Feature Scaling (Standardization, Min-Max Scaling)  
✅ Encoding Categorical Data  



# 📌 Day 4: Exploratory Data Analysis (EDA)  

## 🔹 Why is EDA Important?  
EDA helps us understand patterns in data before applying ML models.  

## 🔹 Key EDA Techniques Covered  
✅ Summary Statistics (`describe()`, `info()`)  
✅ Handling Missing Values (`isnull()`, `fillna()`)  
✅ Data Visualization (Histograms, Boxplots, Correlation Heatmaps)  



# 📌 Day 5: Feature Engineering in Machine Learning  

## 🔹 Why is Feature Engineering Important?  
Feature Engineering transforms raw data into meaningful inputs for ML models.  

## 🔹 Key Techniques Covered  
✅ Handling Missing Values (`SimpleImputer`)  
✅ Encoding Categorical Variables (`OneHotEncoder`)  
✅ Scaling & Normalization (`StandardScaler`)  
✅ Feature Selection (`SelectKBest`)  
✅ Creating New Features (`pd.cut()`)  


# 📌 Day 6: Handling Outliers in Machine Learning (Python)

Outliers are data points that significantly deviate from the rest of the dataset. They can distort machine learning models, reducing accuracy. Detecting and handling them properly improves model performance.

## 📊 **Why Handle Outliers?**
✅ Prevents biased model training  
✅ Enhances data reliability  
✅ Improves prediction accuracy  

## 🔍 **Methods for Outlier Detection and Handling**

### 1️⃣ **Visualization Techniques**
- **Box Plot**: Helps visualize data distribution  
- **Histogram**: Identifies skewness in data  
- **Scatter Plot**: Detects anomalies  

### 2️⃣ **Statistical Methods**
- **Interquartile Range (IQR):** Removes values outside **1.5 times the IQR**  
- **Z-Score:** Filters values with **z-score greater than 3**  

### 3️⃣ **Machine Learning Methods**
- **Isolation Forest**  
- **DBSCAN (Density-Based Clustering)**  

---

# **📌 Day 7: Handling Outliers – Treatment Methods in Machine Learning (Python)**

Outliers can distort statistical measures and negatively impact machine learning models. Once detected (Day 6), the next step is handling them effectively.

## **🔍 Why Handle Outliers?**

✅ Prevents misleading insights

✅ Enhances model stability

✅ Reduces the risk of overfitting

## **🛠️ Methods to Handle Outliers**

1️⃣ *Capping (Winsorization)*

Replaces extreme values with a specified percentile (e.g., 5th and 95th percentiles).
Useful when outliers carry some meaningful information.

2️⃣ *Transformation Techniques*

- Log Transformation: Reduces skewness and minimizes outlier impact.
- Box-Cox Transformation: Normalizes non-Gaussian distributed data.
- Power Transformations: Stabilizes variance.

3️⃣ *Removing Outliers*

- Based on statistical measures (e.g., IQR, Z-score).
- Ideal when outliers are due to data errors or extreme noise.

---

# 🧪🤖 𝐃𝐚𝐲 𝟖: 𝐓𝐫𝐚𝐢𝐧-𝐓𝐞𝐬𝐭 𝐒𝐩𝐥𝐢𝐭𝐭𝐢𝐧𝐠 & 𝐂𝐫𝐨𝐬𝐬-𝐕𝐚𝐥𝐢𝐝𝐚𝐭𝐢𝐨𝐧 𝐢𝐧 𝐌𝐚𝐜𝐡𝐢𝐧𝐞 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠  | 𝟑𝟎-𝐃𝐚𝐲 𝐌𝐋 𝐂𝐡𝐚𝐥𝐥𝐞𝐧𝐠𝐞



## 🔍 𝐖𝐡𝐲 𝐃𝐨 𝐖𝐞 𝐒𝐩𝐥𝐢𝐭 𝐃𝐚𝐭𝐚?

 ✅ Avoid Overfitting – Helps the model generalize to new data.

 ✅ Model Evaluation – Measures performance on unseen data before deployment.

 ✅ Better Decision-Making – Ensures the model isn't biased toward specific data patterns.



## 🛠 𝐓𝐫𝐚𝐢𝐧-𝐓𝐞𝐬𝐭 𝐒𝐩𝐥𝐢𝐭 (𝐁𝐚𝐬𝐢𝐜 𝐌𝐞𝐭𝐡𝐨𝐝)

The dataset is divided into two parts:

 📌 Training Set (80%) – Used to train the model.

 📌 Test Set (20%) – Used to evaluate model performance.

 🔹 random_state=42 ensures reproducibility.

 🔹 test_size=0.2 means 20% of the data is used for testing.



## 🛠 𝐂𝐫𝐨𝐬𝐬-𝐕𝐚𝐥𝐢𝐝𝐚𝐭𝐢𝐨𝐧: 𝐀 𝐌𝐨𝐫𝐞 𝐑𝐞𝐥𝐢𝐚𝐛𝐥𝐞 𝐀𝐩𝐩𝐫𝐨𝐚𝐜𝐡

Instead of a single train-test split, cross-validation divides the dataset into multiple folds, training the model on different subsets and evaluating on the remaining fold. This reduces bias and variance.



### 🔄 𝑲-𝑭𝒐𝒍𝒅 𝑪𝒓𝒐𝒔𝒔-𝑽𝒂𝒍𝒊𝒅𝒂𝒕𝒊𝒐𝒏

K-Fold CV splits the dataset into K equal parts (e.g., 5 or 10).

The model is trained K times, with a different fold used for testing each time.



The final performance is the average of all K evaluations.

 🔹 KFold(n_splits=5): Splits the data into 5 folds.

 🔹 shuffle=True: Randomizes the data before splitting.



## 🚀𝐒𝐮𝐦𝐦𝐚𝐫𝐲 & 𝐊𝐞𝐲 𝐓𝐚𝐤𝐞𝐚𝐰𝐚𝐲𝐬

 ✅ Train-test split is essential for model evaluation.

 ✅ Cross-validation improves reliability by testing on multiple subsets.

 ✅ K-Fold CV (K=5 or 10) is commonly used for robust evaluation.

 ---

 # ⚖ 𝐃𝐚𝐲 𝟗: 𝐅𝐞𝐚𝐭𝐮𝐫𝐞 𝐒𝐜𝐚𝐥𝐢𝐧𝐠 – 𝐍𝐨𝐫𝐦𝐚𝐥𝐢𝐳𝐚𝐭𝐢𝐨𝐧 & 𝐒𝐭𝐚𝐧𝐝𝐚𝐫𝐝𝐢𝐳𝐚𝐭𝐢𝐨𝐧 𝐢𝐧 𝐌𝐚𝐜𝐡𝐢𝐧𝐞 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 | 𝟑𝟎-𝐃𝐚𝐲 𝐌𝐋 𝐂𝐡𝐚𝐥𝐥𝐞𝐧𝐠𝐞



Feature scaling is a crucial preprocessing step in machine learning. Many algorithms perform better when numerical features are on the same scale. Today, we’ll explore Normalization and Standardization—two widely used techniques.



## 🔍 Why Feature Scaling?

     ✅ Improves Model Performance – Some ML algorithms are sensitive to scale differences.

     ✅ Speeds Up Training – Gradient descent converges faster when features are scaled.

     ✅ Enhances Comparability – Keeps all features on a similar range.



## 📌 Normalization (Min-Max Scaling)

Normalization (also called Min-Max Scaling) transforms features to a fixed range, typically [0,1] or [-1,1].



      ✅ Best for neural networks and distance-based models (e.g., KNN, K-Means).

🔹 Transforms values between 0 and 1.

🔹 Sensitive to outliers (can distort scaling).



## 📌 Standardization (Z-Score Scaling)

Standardization (also called Z-score normalization) transforms features to have zero mean and unit variance.



     ✅ Best for algorithms like Logistic Regression, SVM, PCA, and Linear Regression.

🔹 Works well for normally distributed data.

🔹 Less sensitive to outliers than Min-Max Scaling.



## 🚀 When to Use Which?

🔹 Use Normalization if the data follows a non-Gaussian distribution and models like KNN, K-Means, Neural Networks.

🔹 Use Standardization if the data is normally distributed or required by algorithms like SVM, Linear Regression, or PCA.



## 📌 Summary & Key Takeaways

✅ Scaling is crucial for optimal model performance.

✅ Normalization (Min-Max) scales data between [0,1].

✅ Standardization (Z-score) ensures zero mean and unit variance.

✅ Different algorithms prefer different scaling techniques.

---

# 💻𝐃𝐚𝐲 𝟏𝟎: 𝐒𝐮𝐩𝐞𝐫𝐯𝐢𝐬𝐞𝐝 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 – 𝐈𝐧𝐭𝐫𝐨𝐝𝐮𝐜𝐭𝐢𝐨𝐧 𝐭𝐨 𝐑𝐞𝐠𝐫𝐞𝐬𝐬𝐢𝐨𝐧 | 𝟑𝟎-𝐃𝐚𝐲 𝐌𝐋 𝐂𝐡𝐚𝐥𝐥𝐞𝐧𝐠𝐞



Today, we are diving into Supervised Learning, focusing on Regression, one of the foundational techniques in predictive modeling.



## ✅ 𝐖𝐡𝐚𝐭 𝐢𝐬 𝐒𝐮𝐩𝐞𝐫𝐯𝐢𝐬𝐞𝐝 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠?

Supervised learning is a type of machine learning where the model is trained on 𝐥𝐚𝐛𝐞𝐥𝐞𝐝 𝐝𝐚𝐭𝐚 — meaning both 𝐢𝐧𝐩𝐮𝐭 (𝐟𝐞𝐚𝐭𝐮𝐫𝐞𝐬) and 𝐨𝐮𝐭𝐩𝐮𝐭 (𝐭𝐚𝐫𝐠𝐞𝐭) are known.

     📈 Regression is a type of supervised learning that predicts continuous values (e.g., price, temperature, salary).



## 📊 𝐖𝐡𝐚𝐭 𝐢𝐬 𝐑𝐞𝐠𝐫𝐞𝐬𝐬𝐢𝐨𝐧?

Regression models help us:

     ✅ Predict numeric outcomes (continuous target).

     ✅ Understand relationships between variables.

     ✅ Estimate trends and make forecasts.



## 🚀 𝐓𝐲𝐩𝐞𝐬 𝐨𝐟 𝐑𝐞𝐠𝐫𝐞𝐬𝐬𝐢𝐨𝐧 𝐌𝐨𝐝𝐞𝐥𝐬

    1️⃣ Linear Regression – Predict target based on linear relationship.

    2️⃣ Multiple Linear Regression – Multiple input variables for prediction.

    3️⃣ Polynomial Regression – Non-linear relationships.

    4️⃣ Regularized Regression – Ridge, Lasso (to prevent overfitting).



## ✅ Key Insights

  📌 Linear Regression is a simple yet powerful method to model relationships between variables.

  📌 R² score tells us how well the model fits the data (closer to 1 is better).

  📌 Regression line helps visualize predictions.


## 🔑 Summary

Regression predicts continuous outputs using input features.

Linear models assume a straight-line relationship.

Evaluating model using MSE and R² is essential.

---
# 💻Day 11: Linear Regression – Implementation & Evaluation



Linear Regression is a supervised learning algorithm used for predicting continuous numeric values based on input features.

Models relationship between independent (X) and dependent (y) variables.

Predicts outcome using a linear relationship.



## Implementation & Evaluation

 ✅ Implement Linear Regression in Python.

 ✅ Train and evaluate the model.

 ✅ Understand key performance metrics.

 ✅ Visualize the results for better insights.



## ✅ Performance Metrics

Mean Squared Error (MSE): Measures average squared difference between predicted and actual values. Lower is better.

R² Score: Indicates how well data fit the regression model. Closer to 1 means better fit.



-> Linear Regression models a linear relationship between inputs and output.

-> Important to evaluate the model using MSE and R².

-> Visualization of the regression line helps understand fit and trend.

---

# 📊 𝐃𝐚𝐲 𝟏𝟐: 𝐏𝐨𝐥𝐲𝐧𝐨𝐦𝐢𝐚𝐥 𝐑𝐞𝐠𝐫𝐞𝐬𝐬𝐢𝐨𝐧 – 𝐖𝐡𝐞𝐧 𝐭𝐨 𝐔𝐬𝐞 & 𝐇𝐨𝐰 𝐈𝐭 𝐖𝐨𝐫𝐤𝐬 | 𝟑𝟎-𝐃𝐚𝐲 𝐌𝐋 𝐂𝐡𝐚𝐥𝐥𝐞𝐧𝐠𝐞



 Today, let's dive into Polynomial Regression — a powerful method for handling non-linear relationships in data.



## ✅ 𝐖𝐡𝐚𝐭 𝐢𝐬 𝐏𝐨𝐥𝐲𝐧𝐨𝐦𝐢𝐚𝐥 𝐑𝐞𝐠𝐫𝐞𝐬𝐬𝐢𝐨𝐧?

Polynomial Regression is an extension of Linear Regression that models the relationship between independent and dependent variables as an nth-degree polynomial.

It is used when linear models fail to capture complex patterns in data.



## 🔑 𝐖𝐡𝐞𝐧 𝐭𝐨 𝐔𝐬𝐞 𝐏𝐨𝐥𝐲𝐧𝐨𝐦𝐢𝐚𝐥 𝐑𝐞𝐠𝐫𝐞𝐬𝐬𝐢𝐨𝐧?

 ✅ When the relationship between variables is non-linear.

 ✅ When residual plots of linear regression show patterns (sign of underfitting).

 ✅ To model complex curves and trends in the data.


## 📊 𝐊𝐞𝐲 𝐈𝐧𝐬𝐢𝐠𝐡𝐭𝐬

 ⚙️ Polynomial Regression captures non-linear patterns effectively.

 ⚙️ Degree selection is crucial — too low: underfitting, too high: overfitting.

 ⚙️ Always compare with Linear Regression to assess improvement.

 ⚙️ Check R² Score (closer to 1 is better) and MSE (lower is better) to evaluate fit.



 ✅ Polynomial Regression is essential for curved patterns where linear models fail.

 ✅ Balance between model complexity and performance is key — use visualization and metrics to choose degree.
 
 ---
 # 📈 Day 13: Logistic Regression – Binary & Multiclass Classification

## 📌 Overview
Logistic Regression is a popular classification algorithm used for binary and multiclass classification problems. Unlike Linear Regression, it predicts probabilities and uses sigmoid/softmax functions to output values between 0 and 1.


## ✅ Binary Classification Example:
- Dataset: Breast Cancer Dataset (Malignant/Benign)
- Splitting data using train_test_split
- Logistic Regression model fitting
- Accuracy and Classification Report

### Code Output Example:
```
Accuracy Score: 0.956140350877193

Classification Report:
              precision    recall  f1-score   support

           0       0.96      0.96      0.96        42
           1       0.95      0.95      0.95        72

    accuracy                           0.96       114
   macro avg       0.96      0.96      0.96       114
weighted avg       0.96      0.96      0.96       114
```


## ✅ Multiclass Classification Example:
- Dataset: Iris Dataset (3 classes)
- Using multinomial Logistic Regression
- Accuracy and Classification Report

### Code Output Example:
```
Accuracy Score: 1.0

Classification Report:
              precision    recall  f1-score   support

    setosa       1.00      1.00      1.00        10
versicolor       1.00      1.00      1.00         9
 virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```


## 📈 Insights:
- Logistic Regression is powerful for simple classification tasks.
- Binary classification uses sigmoid function.
- Multiclass classification uses softmax (multinomial).
- Easy to implement and interpret.

---

# 📊 𝐃𝐚𝐲 𝟏𝟒: 𝐌𝐨𝐝𝐞𝐥 𝐄𝐯𝐚𝐥𝐮𝐚𝐭𝐢𝐨𝐧 𝐌𝐞𝐭𝐫𝐢𝐜𝐬 – 𝐌𝐀𝐄, 𝐌𝐒𝐄, 𝐑𝐌𝐒𝐄, 𝐀𝐜𝐜𝐮𝐫𝐚𝐜𝐲, 𝐏𝐫𝐞𝐜𝐢𝐬𝐢𝐨𝐧, 𝐑𝐞𝐜𝐚𝐥𝐥 | 𝟑𝟎-𝐃𝐚𝐲 𝐌𝐋 𝐂𝐡𝐚𝐥𝐥𝐞𝐧𝐠𝐞



Evaluating a Machine Learning model is crucial to measure how well it performs on unseen data. Different problems require different evaluation metrics!



## ✅𝐖𝐡𝐲 𝐌𝐨𝐝𝐞𝐥 𝐄𝐯𝐚𝐥𝐮𝐚𝐭𝐢𝐨𝐧 𝐢𝐬 𝐈𝐦𝐩𝐨𝐫𝐭𝐚𝐧𝐭?

     🎯 Assess model's prediction quality

     📈 Compare multiple models

     🔁 Improve model performance through tuning



## 🔑 𝐊𝐞𝐲 𝐄𝐯𝐚𝐥𝐮𝐚𝐭𝐢𝐨𝐧 𝐌𝐞𝐭𝐫𝐢𝐜𝐬:



### 🔵 𝑭𝒐𝒓 𝑹𝒆𝒈𝒓𝒆𝒔𝒔𝒊𝒐𝒏 𝑷𝒓𝒐𝒃𝒍𝒆𝒎𝒔:

1️⃣ 𝗠𝗔𝗘 (𝗠𝗲𝗮𝗻 𝗔𝗯𝘀𝗼𝗹𝘂𝘁𝗲 𝗘𝗿𝗿𝗼𝗿) – Average of absolute errors.

     ➡️ Measures average magnitude of errors.

2️⃣ 𝗠𝗦𝗘 (𝗠𝗲𝗮𝗻 𝗦𝗾𝘂𝗮𝗿𝗲𝗱 𝗘𝗿𝗿𝗼𝗿) – Average of squared errors.

     ➡️ Penalizes larger errors more than smaller ones.

3️⃣ 𝗥𝗠𝗦𝗘 (𝗥𝗼𝗼𝘁 𝗠𝗲𝗮𝗻 𝗦𝗾𝘂𝗮𝗿𝗲𝗱 𝗘𝗿𝗿𝗼𝗿) – Square root of MSE.

     ➡️ Interpretable in same units as target variable.



### 🟢 𝑭𝒐𝒓 𝑪𝒍𝒂𝒔𝒔𝒊𝒇𝒊𝒄𝒂𝒕𝒊𝒐𝒏 𝑷𝒓𝒐𝒃𝒍𝒆𝒎𝒔:

4️⃣ 𝗔𝗰𝗰𝘂𝗿𝗮𝗰𝘆 – Proportion of correctly classified samples.

      ➡️ Simple but can be misleading for imbalanced datasets.

5️⃣ 𝗣𝗿𝗲𝗰𝗶𝘀𝗶𝗼𝗻 – True Positives / (True Positives + False Positives).

      ➡️ How many predicted positives are actually positive.

6️⃣ 𝗥𝗲𝗰𝗮𝗹𝗹 – True Positives / (True Positives + False Negatives).

      ➡️ Ability to find all actual positives.



## ✅ 𝑰𝒏𝒔𝒊𝒈𝒉𝒕𝒔:

      📊 Use MAE, MSE, RMSE for regression to measure error magnitude.

      🧠 Accuracy is good for balanced datasets, but Precision & Recall are better for imbalanced ones (like fraud detection, medical diagnosis).

      🔍 Always choose metric based on business problem.

---

#  🌳🤖 Day 15 — Decision Trees in Machine Learning 

Decision Trees, a powerful and intuitive algorithm for both classification and regression tasks.

## 🌳 What is a Decision Tree?
A Decision Tree is a flowchart-like structure where:

- Each internal node represents a decision on a feature.

- Each branch represents an outcome of that decision.

- Each leaf node represents a final output label (class/number).

✅ Easy to interpret

✅ Handles numerical & categorical data

✅ Non-linear relationships

## 💡 How does it work?
- Splits data based on the feature that best separates the classes/values.
Uses criteria like:
  - Gini Index or Entropy (Information Gain) for classification.
  - Mean Squared Error (MSE) for regression.
## 📊 Applications
- Classification (e.g., Spam Detection, Loan Approval)
- Regression (e.g., House Price Prediction)
- Feature Importance Ranking

---

# 🌲✨ Day 16: Random Forest – Bagging & Boosting Techniques 

Today, we're diving into Random Forest, one of the most powerful and widely used ensemble learning algorithms. We'll explore Bagging (Bootstrap Aggregating) and touch on Boosting, understanding how these methods enhance model performance! 💪📊

## 🌟 What is Random Forest?

✅ An ensemble learning method that builds multiple decision trees and merges their outputs for better accuracy and control over overfitting.

✅ Works for both classification and regression problems.

✅ Based on Bagging (Bootstrap Aggregating) technique.

## 💡 Why Random Forest?
🚀 Handles large datasets with higher dimensionality.

🚀 Reduces overfitting by averaging multiple trees.

🚀 Improves accuracy compared to a single decision tree.

🚀 Handles missing values and maintains accuracy for missing data.


## 🔑 Key Concepts:
🔹 Bagging:

- Random sampling of data with replacement.

- Multiple models trained in parallel.
    
- Averaging (regression) or voting (classification) for final output.
    
- Goal: Reduce variance & prevent overfitting.

🔹 Boosting (Brief Intro):

- Models trained sequentially.
    
- Each new model focuses on correcting the previous model's mistakes.
    
- Algorithms: AdaBoost, Gradient Boosting, XGBoost.
    
- Goal: Reduce bias & improve prediction strength.

---

## 📌 Day 17: Support Vector Machines (SVM) – Concept & Implementation

Today, I explored one of the most powerful algorithms in Machine Learning — Support Vector Machines (SVM)! 🚀

## 🔑 What is SVM?
- SVM is a supervised learning algorithm used for classification and regression.
- It works by finding the optimal hyperplane that separates data points of different classes with maximum margin.
- SVM is highly effective in high-dimensional spaces and non-linear datasets when used with kernel tricks.

## ✅ What I covered today:

### 1️⃣ SVM Classification
- Dataset: Iris Dataset 🌸
- Used Linear & Non-linear (RBF Kernel) SVM for classification.
- Visualized decision boundaries and support vectors.
- Achieved high accuracy and interpreted results effectively.
- 📊 Visualization of Decision Boundary helps in understanding the separation of classes.

### 2️⃣ SVM Regression (SVR)
- Dataset: California Housing Dataset 🏘️
- Used Support Vector Regressor (SVR) to predict housing prices.
- Evaluated using MSE, RMSE, and R² score for performance assessment.

## 🔍 Key Takeaways:
- SVM performs exceptionally well in complex datasets with proper tuning.
- Kernel Trick (like RBF, Polynomial) enables SVM to handle non-linear classification problems.
- Support Vectors play a crucial role in defining the hyperplane and margin.
- SVR is an effective technique for regression problems where robustness to outliers is needed.

---

# 🚀 Day18: k-Nearest Neighbors (k-NN) – Classification & Regression

k-NN is a powerful, yet simple algorithm used for **both classification and regression**. It makes predictions based on the **majority vote** of k-nearest neighbors in the feature space.

## 🔹 **Key Features:**
✅ Works well with both **classification & regression**.

✅ Uses **distance measures** like Euclidean, Manhattan, etc.

✅ **Decision boundary visualization** helps understand how k-NN classifies data.

## 🛠 **Implementation Highlights:**
- 🔸 k-NN **Classification** – Predicts classes based on **majority voting**.
- 🔸 k-NN **Regression** – Predicts values by averaging **k-nearest neighbors**.
- 🔸 **Visualization of Decision Boundaries & Predictions** included.

## 📊 **Results:**
- ✔️ **Accuracy (Classification):** 91%
- ✔️ **MSE (Regression):** 0.62, **R² Score:** 0.75

## 🔍 **Key Insights:**
- The choice of **k** affects model performance.
- **Distance metric (Euclidean/Manhattan)** plays a crucial role.
- **k-NN is simple yet effective** for many ML applications.

---

# 📌 𝐃𝐚𝐲 𝟏𝟗: 𝐍𝐚ï𝐯𝐞 𝐁𝐚𝐲𝐞𝐬 – 𝐏𝐫𝐨𝐛𝐚𝐛𝐢𝐥𝐢𝐬𝐭𝐢𝐜 𝐂𝐥𝐚𝐬𝐬𝐢𝐟𝐢𝐞𝐫 & 𝐔𝐬𝐞 𝐂𝐚𝐬𝐞𝐬



Naïve Bayes is a powerful probabilistic classifier based on Bayes’ theorem, assuming that features are independent (hence "naïve"). It’s widely used in spam detection, sentiment analysis, medical diagnosis, and more!



## 🔹 𝐓𝐲𝐩𝐞𝐬 𝐨𝐟 𝐍𝐚ï𝐯𝐞 𝐁𝐚𝐲𝐞𝐬 𝐂𝐥𝐚𝐬𝐬𝐢𝐟𝐢𝐞𝐫𝐬

 ✔ Gaussian Naïve Bayes (GNB): Assumes normal distribution of features.

 ✔ Multinomial Naïve Bayes (MNB): Best for text classification (e.g., spam filtering).

 ✔ Bernoulli Naïve Bayes (BNB): Used for binary features (e.g., word presence in a document).



## 🔹 𝐊𝐞𝐲 𝐓𝐚𝐤𝐞𝐚𝐰𝐚𝐲𝐬

 ✅ Fast & efficient for large datasets

 ✅ Performs well with small datasets & categorical data

 ✅ Great for NLP tasks (spam detection, sentiment analysis)

---

# 🚀 𝐃𝐚𝐲 𝟐𝟎: 𝐇𝐚𝐧𝐝𝐥𝐢𝐧𝐠 𝐈𝐦𝐛𝐚𝐥𝐚𝐧𝐜𝐞𝐝 𝐃𝐚𝐭𝐚 – 𝐒𝐌𝐎𝐓𝐄, 𝐂𝐥𝐚𝐬𝐬 𝐖𝐞𝐢𝐠𝐡𝐭𝐬, 𝐓𝐡𝐫𝐞𝐬𝐡𝐨𝐥𝐝𝐢𝐧𝐠 | 𝟑𝟎-𝐃𝐚𝐲 𝐌𝐋 𝐂𝐡𝐚𝐥𝐥𝐞𝐧𝐠𝐞



 Many real-world datasets have an unequal distribution of classes, leading to biased models. We explore three techniques to handle class imbalance:



 ✅ SMOTE (Synthetic Minority Over-sampling Technique) – Creates synthetic samples for the minority class.

 ✅ Class Weights – Assigns higher weights to the minority class to balance training.

 ✅ Thresholding – Adjusts the decision threshold to optimize model performance.



## 🔹 𝐊𝐞𝐲 𝐓𝐚𝐤𝐞𝐚𝐰𝐚𝐲𝐬

 ✅ SMOTE creates synthetic data to balance the dataset.

 ✅ Class Weights make the model more sensitive to the minority class.

 ✅ Thresholding helps control precision and recall trade-offs.

---

# 🔍 𝐃𝐚𝐲 𝟐𝟏: 𝐇𝐲𝐩𝐞𝐫𝐩𝐚𝐫𝐚𝐦𝐞𝐭𝐞𝐫 𝐓𝐮𝐧𝐢𝐧𝐠 – 𝐆𝐫𝐢𝐝 𝐒𝐞𝐚𝐫𝐜𝐡, 𝐑𝐚𝐧𝐝𝐨𝐦 𝐒𝐞𝐚𝐫𝐜𝐡, 𝐁𝐚𝐲𝐞𝐬𝐢𝐚𝐧 𝐎𝐩𝐭𝐢𝐦𝐢𝐳𝐚𝐭𝐢𝐨𝐧 | 𝟑𝟎-𝐃𝐚𝐲 𝐌𝐋 𝐂𝐡𝐚𝐥𝐥𝐞𝐧𝐠𝐞



Hyperparameter tuning is the key to unlocking the full potential of machine learning models. Choosing the right method can significantly impact model performance, training efficiency, and computational cost.

Here’s a breakdown of three powerful tuning techniques:



## 1️⃣ 𝐆𝐫𝐢𝐝 𝐒𝐞𝐚𝐫𝐜𝐡 – 𝐄𝐱𝐡𝐚𝐮𝐬𝐭𝐢𝐯𝐞 𝐛𝐮𝐭 𝐂𝐨𝐬𝐭𝐥𝐲

 ✅ Searches all possible hyperparameter combinations.

 ✅ Best for small search spaces.

 ⚠️ Computationally expensive for large datasets.



## 2️⃣ 𝐑𝐚𝐧𝐝𝐨𝐦 𝐒𝐞𝐚𝐫𝐜𝐡 – 𝐅𝐚𝐬𝐭𝐞𝐫 & 𝐄𝐟𝐟𝐢𝐜𝐢𝐞𝐧𝐭

 ✅ Randomly selects hyperparameter combinations.

 ✅ Balances speed and accuracy well.

 ⚠️ May miss the best hyperparameters but often finds near-optimal ones.



## 3️⃣ 𝐁𝐚𝐲𝐞𝐬𝐢𝐚𝐧 𝐎𝐩𝐭𝐢𝐦𝐢𝐳𝐚𝐭𝐢𝐨𝐧 – 𝐒𝐦𝐚𝐫𝐭 & 𝐀𝐝𝐚𝐩𝐭𝐢𝐯𝐞

 ✅ Uses probability-based methods to find optimal hyperparameters.

 ✅ Works well for large and complex search spaces.

 ✅ Faster than Grid & Random Search in many cases.



## 💡 𝐖𝐡𝐢𝐜𝐡 𝐨𝐧𝐞 𝐬𝐡𝐨𝐮𝐥𝐝 𝐲𝐨𝐮 𝐮𝐬𝐞?

- If your dataset is small, Grid Search can work.

- For moderate datasets, Random Search is a great balance.

- When dealing with large-scale ML problems, Bayesian Optimization is a game-changer.

- Efficient hyperparameter tuning can lead to higher model accuracy, reduced overfitting, and faster training times. 🚀

---

# 📈 𝐃𝐚𝐲 𝟐𝟐: 𝐈𝐧𝐭𝐫𝐨𝐝𝐮𝐜𝐭𝐢𝐨𝐧 𝐭𝐨 𝐂𝐥𝐮𝐬𝐭𝐞𝐫𝐢𝐧𝐠 – 𝐤-𝐌𝐞𝐚𝐧𝐬 & 𝐇𝐢𝐞𝐫𝐚𝐫𝐜𝐡𝐢𝐜𝐚𝐥 𝐂𝐥𝐮𝐬𝐭𝐞𝐫𝐢𝐧𝐠 | 𝟑𝟎-𝐃𝐚𝐲 𝐌𝐋 𝐂𝐡𝐚𝐥𝐥𝐞𝐧𝐠𝐞



Clustering is a fundamental unsupervised machine learning technique used to group similar data points together. 



## 𝗧𝘄𝗼 𝗽𝗼𝗽𝘂𝗹𝗮𝗿 𝗰𝗹𝘂𝘀𝘁𝗲𝗿𝗶𝗻𝗴 𝗮𝗹𝗴𝗼𝗿𝗶𝘁𝗵𝗺𝘀:

 🔹 k-Means Clustering – A centroid-based method that partitions data into k clusters.

 🔹 Hierarchical Clustering – A tree-based clustering method that creates a hierarchy of clusters.



## 𝗪𝗵𝘆 𝗖𝗹𝘂𝘀𝘁𝗲𝗿𝗶𝗻𝗴?

 ✅ Identifies hidden patterns in data.

 ✅ Helps in customer segmentation, anomaly detection, and image segmentation.

 ✅ Used in market research, bioinformatics, and recommendation systems.



## 📊 𝗩𝗶𝘀𝘂𝗮𝗹𝗶𝘇𝗶𝗻𝗴 𝗸-𝗠𝗲𝗮𝗻𝘀 & 𝗛𝗶𝗲𝗿𝗮𝗿𝗰𝗵𝗶𝗰𝗮𝗹 𝗖𝗹𝘂𝘀𝘁𝗲𝗿𝗶𝗻𝗴

 🔹 k-Means: Assigns data points to clusters based on proximity to centroids.

 🔹 Dendrogram (Hierarchical Clustering): Illustrates how clusters merge at different levels.



## 🚀 𝗞𝗲𝘆 𝗧𝗮𝗸𝗲𝗮𝘄𝗮𝘆𝘀:

 🔹 k-Means is efficient but requires pre-defining the number of clusters.

 🔹 Hierarchical Clustering does not require specifying k but is computationally expensive for large datasets.

 🔹 Both methods are widely used for data exploration and pattern recognition.

---

# 👩‍💻 𝐃𝐚𝐲 𝟐𝟑: 𝐃𝐁𝐒𝐂𝐀𝐍 – 𝐃𝐞𝐧𝐬𝐢𝐭𝐲-𝐁𝐚𝐬𝐞𝐝 𝐂𝐥𝐮𝐬𝐭𝐞𝐫𝐢𝐧𝐠 𝐟𝐨𝐫 𝐀𝐧𝐨𝐦𝐚𝐥𝐲 𝐃𝐞𝐭𝐞𝐜𝐭𝐢𝐨𝐧 | 𝟑𝟎-𝐃𝐚𝐲 𝐌𝐋 𝐂𝐡𝐚𝐥𝐥𝐞𝐧𝐠𝐞

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a powerful clustering algorithm that identifies dense regions in data and separates them from noise (outliers). Unlike k-Means, it does not require predefining the number of clusters and can detect clusters of arbitrary shapes.

## 🚀 𝐖𝐡𝐲 𝐃𝐁𝐒𝐂𝐀𝐍?
 ✅ Detects anomalies/outliers in datasets.

 ✅ Handles clusters of different densities & shapes effectively.
 
 ✅ Works well for large datasets with noise.

## 📊 𝐇𝐨𝐰 𝐃𝐁𝐒𝐂𝐀𝐍 𝐖𝐨𝐫𝐤𝐬?
 1️⃣ Defines core points with minimum neighbors (MinPts) within a given radius (ε).

 2️⃣ Expands clusters from core points while marking noise/outliers.

 3️⃣ Assigns remaining points to the nearest cluster.

## 🔍 𝐊𝐞𝐲 𝐓𝐚𝐤𝐞𝐚𝐰𝐚𝐲𝐬:
 🔹 Handles noise and anomalies better than k-Means.

 🔹 No need to specify number of clusters beforehand.

 🔹 Works well for geospatial data, anomaly detection, and customer segmentation.

 ---
 
 # 💻 𝐃𝐚𝐲 𝟐𝟒: 𝐏𝐫𝐢𝐧𝐜𝐢𝐩𝐚𝐥 𝐂𝐨𝐦𝐩𝐨𝐧𝐞𝐧𝐭 𝐀𝐧𝐚𝐥𝐲𝐬𝐢𝐬 (𝐏𝐂𝐀) – 𝐃𝐢𝐦𝐞𝐧𝐬𝐢𝐨𝐧𝐚𝐥𝐢𝐭𝐲 𝐑𝐞𝐝𝐮𝐜𝐭𝐢𝐨𝐧 | 𝟑𝟎-𝐃𝐚𝐲 𝐌𝐋 𝐂𝐡𝐚𝐥𝐥𝐞𝐧𝐠𝐞



## 🔎 𝐖𝐡𝐚𝐭 𝐢𝐬 𝐏𝐂𝐀?

 Principal Component Analysis (PCA) is a powerful dimensionality reduction technique used to transform high-dimensional data into a lower-dimensional space while preserving maximum variance.



## 🚀 𝐖𝐡𝐲 𝐏𝐂𝐀?

 ✅ Reduces computational cost for machine learning models.

 ✅ Helps in visualizing high-dimensional data.

 ✅ Removes correlation and redundancy in features.

 ✅ Enhances model performance by avoiding the curse of dimensionality.



## 📊 𝐇𝐨𝐰 𝐏𝐂𝐀 𝐖𝐨𝐫𝐤𝐬?

 1️⃣ Standardize the data.

 2️⃣ Compute the covariance matrix.

 3️⃣ Find the eigenvalues & eigenvectors.

 4️⃣ Select top principal components based on explained variance.

 5️⃣ Project data onto the new feature space.



## 🔍 𝐊𝐞𝐲 𝐓𝐚𝐤𝐞𝐚𝐰𝐚𝐲𝐬:

 🔹 Dimensionality Reduction without significant data loss.

 🔹 Helps in feature selection & noise reduction.

 🔹 Essential for high-dimensional datasets like images, finance, and genomics.

---

 # 🛒📊𝐃𝐚𝐲 𝟐𝟓: 𝐀𝐬𝐬𝐨𝐜𝐢𝐚𝐭𝐢𝐨𝐧 𝐑𝐮𝐥𝐞 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 – 𝐀𝐩𝐫𝐢𝐨𝐫𝐢 & 𝐌𝐚𝐫𝐤𝐞𝐭 𝐁𝐚𝐬𝐤𝐞𝐭 𝐀𝐧𝐚𝐥𝐲𝐬𝐢𝐬 | 𝟑𝟎-𝐃𝐚𝐲 𝐌𝐋 𝐂𝐡𝐚𝐥𝐥𝐞𝐧𝐠𝐞 



In the world of data-driven decision-making, understanding customer behavior is key! 

🔑 Association Rule Learning helps uncover hidden relationships between items, making it invaluable for applications like market basket analysis, recommendation systems, and fraud detection.



## 🔍 𝐖𝐡𝐚𝐭 𝐢𝐬 𝐀𝐬𝐬𝐨𝐜𝐢𝐚𝐭𝐢𝐨𝐧 𝐑𝐮𝐥𝐞 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠?

It’s a technique used to find patterns and correlations in large datasets. The two key concepts are:

 📌 Support – How frequently an itemset appears in transactions.

 📌 Confidence – The likelihood that one item appears given another.

 📌 Lift – The strength of the association compared to random chance.



## 🏆 𝐀𝐩𝐫𝐢𝐨𝐫𝐢 𝐀𝐥𝐠𝐨𝐫𝐢𝐭𝐡𝐦

Apriori is an efficient algorithm that iteratively finds frequent itemsets and derives association rules. It’s widely used in retail, e-commerce, and healthcare for insights like:

 ✅ "People who buy bread also buy butter!" 🥖🧈

 ✅ "Customers who purchase laptops often buy accessories!" 💻🎧



## 🔥 𝐌𝐚𝐫𝐤𝐞𝐭 𝐁𝐚𝐬𝐤𝐞𝐭 𝐀𝐧𝐚𝐥𝐲𝐬𝐢𝐬 𝐢𝐧 𝐀𝐜𝐭𝐢𝐨𝐧

 🔹 Used by Amazon, Walmart, and Netflix to boost cross-selling.

 🔹 Helps banks detect fraudulent transactions.

 🔹 Optimizes product placement in supermarkets for increased sales.


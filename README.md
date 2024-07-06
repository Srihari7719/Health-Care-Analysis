# Healthcare Dataset

## Introduction 
This synthetic healthcare dataset has been created to serve as a valuable resource for data science, machine learning, and data analysis enthusiasts. It is designed to mimic real-world healthcare data, enabling users to practice, develop, and showcase their data manipulation and analysis skills in the context of the healthcare industry.

## Dataset


-	**Name**               -  This column represents the name of the patient associated with the healthcare record.  
-	**Age**                -  The age of the patient at the time of admission, expressed in years.
-	**Gender**             -  Indicates the gender of the patient, either "Male" or "Female."
-	**Blood Type**         -  The patient's blood type, which can be one of the common blood types (e.g., "A+", "O-", etc.).
-	**Medical Condition**  - This column specifies the primary medical condition or diagnosis associated with the <br>
 patient, such as "Diabetes," "Hypertension," "Asthma," and more.
-	**Date of Admission**  -  The date on which the patient was admitted to the healthcare facility.
-	**Doctor**             -  The name of the doctor responsible for the patient's care during their admission.
-	**Hospital**           -  Identifies the healthcare facility or hospital where the patient was admitted.
-	**Insurance Provider** -  This column indicates the patient's insurance provider, which can be one of several options, including "Aetna," "Blue Cross," "Cigna," "UnitedHealthcare," and "Medicare."
-   **Billing Amount**     -  The amount of money billed for the patient's healthcare services during their admission. This is expressed as a floating-point number
-	**Room Number**        -  The room number where the patient was accommodated during their admission.
-	**Admission Type**     -  Specifies the type of admission, which can be "Emergency," "Elective," or "Urgent," reflecting the circumstances of the admission.
-	**Discharge Date**     -  The date on which the patient was discharged from the healthcare facility, based on the <br>
 admission date and a random number of days within a realistic range.
-	**Medication**         -  Identifies a medication prescribed or administered to the patient during their admission. <br>
 Examples include "Aspirin," "Ibuprofen," "Penicillin," "Paracetamol," and "Lipitor."
-	**Test results**       -   Describes the results of a medical test conducted during the patient's admission. Possible <br>
values include "Normal," "Abnormal," or "Inconclusive," indicating the outcome of the test.

## Outline

#### This dataset can be utilized for a wide range of purposes, including:

- Developing and testing healthcare predictive models.
- Practicing data cleaning, transformation, and analysis techniques.
- Creating data visualizations to gain insights into healthcare trends.
- Learning and teaching data science and machine learning concepts in a healthcare context.
- You can treat it as a Multi-Class Classification Problem and solve it for Test Results which contains 3 categories(Normal,     
 Abnormal, and Inconclusive).

## Process overview
-Throughout the project, I followed a structured process to develop a predictive model for patient discharge outcomes. Here's an   overview of the process, along with some insights into my experience.
1.	Data Collection and Exploration
2.  Data Preprocessing
3.  Model Selection
4.  Model Training and Evaluation
5.  Iterative Improvement
6.	Validation and Testing
7.	Deployment and Monitoring
## Importing libraries.
```
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
```

```
df = pd.read_csv(r"C:\Users\veerapsh\Downloads\healthcare_dataset.csv")
df.head()
```

<img width="1337" alt="image-1" src="https://github.com/Srihari2811/Ml/assets/103255536/ee38424b-5193-48fd-a48f-9e0239d07a70">

```
df.shape
```
```
Output : 
(10000, 15)
```

```
df.isna().sum()
```
```
Output : 
Name                  0
Age                   0
Gender                0
Blood Type            0
Medical Condition     0
Date of Admission     0
Doctor                0
Hospital              0
Insurance Provider    0
Billing Amount        0
Room Number           0
Admission Type        0
Discharge Date        0
Medication            0
Test Results          0
dtype: int64
```

```
import pandas as pd


# Convert 'Date of Admission' and 'Discharge Date' columns to datetime objects
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])

# Calculate the time difference (length of stay) in days
df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

# Optionally, you can also calculate the time difference in hours, minutes, etc.
# df['Length of Stay (hours)'] = (df['Discharge Date'] - df['Date of Admission']).dt.total_seconds() / 3600

df
```

<img width="1334" alt="image-2" src="https://github.com/Srihari2811/Ml/assets/103255536/82fac826-032a-406d-8ddb-dd24f51772bf">

```
unique_test_results = df['Test Results'].unique()
print("Unique test results:", unique_test_results)
```

```
Output :
Unique test results: ['Inconclusive' 'Normal' 'Abnormal']
```

```
import pandas as pd
inconclusive_proportion = df['Test Results'].value_counts(normalize=True)['Inconclusive']
print("Proportion of 'Inconclusive' cases:", inconclusive_proportion)
Compare characteristics and distributions before and after removing "Inconclusive" cases
data_integrity_before = df.describe()
df_filtered = df[df['Test Results'] != 'Inconclusive']
data_integrity_after = df_filtered.describe()
print("\nData Integrity Before Removing 'Inconclusive' Cases:")
print(data_integrity_before)
print("\nData Integrity After Removing 'Inconclusive' Cases:")
print(data_integrity_after)
```

```
Output :
Proportion of 'Inconclusive' cases: 0.3277

Data Integrity Before Removing 'Inconclusive' Cases:
                Age              Date of Admission  Billing Amount  \
count  10000.000000                          10000    10000.000000   
mean      51.452200  2021-05-01 21:53:25.439999744    25516.806778   
min       18.000000            2018-10-30 00:00:00     1000.180837   
25%       35.000000            2020-02-10 00:00:00    13506.523967   
50%       52.000000            2021-05-02 00:00:00    25258.112566   
75%       68.000000            2022-07-23 06:00:00    37733.913727   
max       85.000000            2023-10-30 00:00:00    49995.902283   
std       19.588974                            NaN    14067.292709   

        Room Number              Discharge Date  Length of Stay  
count  10000.000000                       10000    10000.000000  
mean     300.082000  2021-05-17 11:22:24.960000       15.561800  
min      101.000000         2018-11-01 00:00:00        1.000000  
25%      199.000000         2020-02-23 18:00:00        8.000000  
50%      299.000000         2021-05-18 00:00:00       16.000000  
75%      400.000000         2022-08-07 00:00:00       23.000000  
max      500.000000         2023-11-27 00:00:00       30.000000  
std      115.806027                         NaN        8.612038  

Data Integrity After Removing 'Inconclusive' Cases:
               Age              Date of Admission  Billing Amount  \
count  6723.000000                           6723     6723.000000   
mean     51.578759  2021-05-02 03:56:15.100401408    25438.356013   
min      18.000000            2018-10-30 00:00:00     1000.180837   
25%      35.000000            2020-02-01 00:00:00    13412.992875   
50%      52.000000            2021-05-05 00:00:00    25160.410307   
75%      69.000000            2022-07-27 00:00:00    37632.710076   
max      85.000000            2023-10-30 00:00:00    49995.902283   
std      19.607617                            NaN    14022.117088   

       Room Number                 Discharge Date  Length of Stay  
count  6723.000000                           6723     6723.000000  
mean    300.709653  2021-05-17 18:27:47.469879552       15.605236  
min     101.000000            2018-11-01 00:00:00        1.000000  
25%     199.000000            2020-02-17 00:00:00        8.000000  
50%     300.000000            2021-05-20 00:00:00       16.000000  
75%     401.000000            2022-08-12 00:00:00       23.000000  
max     500.000000            2023-11-24 00:00:00       30.000000  
std     116.038567                            NaN        8.596533  
```

## EDA
### Pair plot of numerical variables

```
    numeric_features = ['Age', 'Billing Amount']<br>
    sns.pairplot(df[numeric_features])<br>
    plt.show()
```
Output:

![image-3](https://github.com/Srihari2811/Ml/assets/103255536/92002e83-60d1-4f0e-ba47-419603f85eb0)

- creates a scatter plot (pair plot) that visually represents the relationship between 'Age' and 'Billing Amount' variables <br>
 from the dataset.The graph shows a positive correlation between age and billing amount. <br> This means that as people get older, their billing amount tends to increase.

### Histogram of Age
```
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Age'], bins=20, kde=True) 
    plt.xlabel('Age') 
    plt.ylabel('Count') 
    plt.title('Distribution of Age') 
    plt.show()
```
Output:

![image-4](https://github.com/Srihari2811/Ml/assets/103255536/7528948b-d626-4ef3-a6af-60afa3902b23)


- The histogram plot to visualize the distribution of 'Age' in the dataset. <br>It follows principles of information visualization 
 by  selecting appropriate plot types, labeling axes, and providing a title for context.<br> This line graph is a simple and effective way to visualize the distribution of age in a population.


### Distribution of Billing Amount by Admission Type


```    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Admission Type', y='Billing Amount', data=df)
    plt.xlabel('Admission Type')
    plt.ylabel('Billing Amount')
    plt.title('Billing Amount by Admission Type')
    plt.show()
```

Output:

![image-5](https://github.com/Srihari2811/Ml/assets/103255536/aec53381-fbaf-4220-868e-d69ef16f6e42)

- The box plot to visualize the distribution of billing amounts across different admission types.<br> It follows principles of information visualization by selecting appropriate plot types, encoding data attributes, labeling axes, and providing a title for context.
-  Analyzing the distribution of 'Billing Amount' across different 'Admission Types' can provide insights into the importance of  'Admission Type' as a predictor for 'Billing Amount'
- The box plot allows identify potential differences in billing amounts across different admission types and detect any outliers

# Feature importance


``` 
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
    df[[ 'Length of Stay']]
```

Output:

<img width="220" alt="image-6" src="https://github.com/Srihari2811/Ml/assets/103255536/de03a0ba-1ee6-4961-aea8-2908d6cb0a27">

- The 'Length of Stay' feature is derived from the difference in between the 'Date of Admission' and 'Discharge Date' columns.<br>
- In machine learning, feature engineering involves creating new features from existing ones to improve model performance.<br>  
- Adding the length of stay as a feature may provide valuable information to predictive models, especially in healthcare-related applications where the duration of hospitalization can be indicative of various outcomes or conditions.

```
    bins = [0, 30, 50, 70, 100]  
    labels = ['Under 30', '31-50', '51-70', 'Over 70']  

    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    df[['Age', 'Age Group']]
```
Output:

<img width="224" alt="image-7" src="https://github.com/Srihari2811/Ml/assets/103255536/b8e76b0b-ff0a-44b8-9070-719e3ea4b08a">

- The 'Age Group' feature transforms the continuous 'Age' feature into a categorical variable.<br> 
- This transformation can capture non-linear relationships between age and the target variable, which may improve the performance of machine learning models that work better with categorical data.


``` 
    average_billing_per_doctor = df.groupby('Doctor')['Billing Amount'].mean()
    print(average_billing_per_doctor)
```

Output:

Doctor
Aaron Barrera       17930.808495
Aaron Brewer        38698.632541
Aaron Brown         19455.289654
Aaron Burnett       37421.296660
Aaron Cameron MD    20216.305804
                        ...     
Zachary Turner      46295.820395
Zachary Walker      17528.856491
Zachary Wong        13730.432921
Zoe Cunningham       4074.530633
Zoe Garza           38099.997517
Name: Billing Amount, Length: 6453, dtype: float64

- By grouping observations based on the 'Doctor' column. This creates distinct groups for each doctor in the dataset.
- Computing the mean of the 'Billing Amount' within each group,we make a new feature that represents the average billing amount associated with each doctor.
- Understanding the average billing amount per doctor can provide insights into the behavior or performance of individual doctors.


```
    age_bins = [0, 20, 40, 60, 80, float('inf')]
    age_labels = ['0-20', '21-40', '41-60', '61-80', '81+']
    df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    df
```

Output:

<img width="1313" alt="image-8" src="https://github.com/Srihari2811/Ml/assets/103255536/9ad8485f-6a75-4ecc-9d55-25de9fff1dcd">

- The continuous 'Age' feature into five age groups: '0-20', '21-40', '41-60', '61-80', and '81+'.<br> This creates a categorical
  variable 'Age Group' that represents different age ranges.
- converting the continuous 'Age' feature into discrete age groups, transforms the data in a way that may be more suitable for certain types of models, such as decision trees or logistic regression, which handle categorical variables more effectively.
- Discretizing age into meaningful categories can potentially improve model performance.



```
    variables_to_remove = ['Name','Date of Admission','Hospital','Room Number','Discharge Date','Doctor','Age']

    df = df.drop(variables_to_remove, axis=1)
    df
```

Output:

<img width="1279" alt="image-9" src="https://github.com/Srihari2811/Ml/assets/103255536/cf9e8404-ebbb-4410-860e-213640b8554c">

- The name, hospital, room number dosent affect the target variable.
- Then the difference between the date of admission and discharge date  created new variable, so were  removed these variables.
- Age variable is created into groups of ages, it was also removed.





# Feature engineering 

### Distribution of Age Groups in the Dataset


``` 
    from sklearn.preprocessing import LabelEncoder
        lc = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col]=lc.fit_transform(df[col])

    df['Age Group'] = lc.fit_transform(df['Age Group'])
    df.head()
```

Output:

<img width="1260" alt="image-10" src="https://github.com/Srihari2811/Ml/assets/103255536/3993fee3-56fe-482b-99ce-ca88d0d64a10">



Here we are performing label encoding,
- Label Encoding is a technique used to convert categorical data into numerical form, which can be more easily processed by machine learning algorithms.
- We perform label encoding on Gender, Blood Type,	Medical Condition, Insurance Provider, Admission Type,	Medication,	Test Result
these variables.



### Correlation Heatmap of Features

```
    import seaborn as sns
    import matplotlib.pyplot as plt

    corr_matrix = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()
```

Output:

![image-11](https://github.com/Srihari2811/Ml/assets/103255536/732649a5-15ab-4bc3-8c27-060ddd746f21)

- Correlation measures the strength and direction of a linear relationship between two variables. 
- There's a strong positive correlation between "Billing Amount" and "Length of Stay" and a weak negative correlation between "Age Group" and "Length of Stay".
- Features in a model that predicts hospital costs, while excluding "Age Group" if it has a weak correlation.



# Model Fitting


## Train test spliting 
```
    X,y=df.drop(['Test Results'],axis=1), df['Test Results']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

```

```
Output :
((5378, 9), (1345,))
```
- Here we perform oparation on  data by splitting it into training and testing sets, which are essential for building and evaluating machine learning models.


```
    X_train.shape, y_test.shape
```
- The size and shape of the training and testing datasets, which is essential for proper data splitting and model training

## Random forest classifier

```
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)   

    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
```

```
Output :
Accuracy: 0.5070631970260223
Classification Report:
              precision    recall  f1-score   support

           0       0.51      0.57      0.54       680
           1       0.50      0.44      0.47       665

    accuracy                           0.51      1345
   macro avg       0.51      0.51      0.50      1345
weighted avg       0.51      0.51      0.50      1345

```


- Accuracy of the model on the testing data is approximately 0.507, which means the model correctly predicts the target variable for about 50.07% of the instances in the testing set.
-  The average of metrics across all classes, weighted equally.
- Model's accuracy is  50%, indicating that its predictive performance.
-  Random Forest model has been trained and evaluated, its performance is modest.



 
## XGBoost Classifier model


```
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
```

```
Output :
Accuracy: 0.50
              precision    recall  f1-score   support

           0       0.51      0.54      0.52       680
           1       0.50      0.46      0.48       665

    accuracy                           0.50      1345
   macro avg       0.50      0.50      0.50      1345
weighted avg       0.50      0.50      0.50      1345

[[367 313]
 [356 309]]
```


- Evaluation metrics include accuracy score, classification report, and confusion matrix.
- The model achieved an accuracy of approximately on the testing data.
- The accuracy score indicates the overall predictive performance of the model on the testing data.
- Precision, recall, and F1-score offer a detailed assessment of the model's ability to classify instances for each class.
- The confusion matrix provides additional context on the distribution of correct and incorrect predictions.
- The achived accuracy was 0.50 for XGBoost Classifier


## Logistic regression model

```
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report


    logistic_regression = LogisticRegression()

    logistic_regression.fit(X_train, y_train)

    y_pred = logistic_regression.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
```

```
Output :
Accuracy: 0.5003717472118959
Classification Report:
              precision    recall  f1-score   support

           0       0.50      0.80      0.62       680
           1       0.49      0.20      0.28       665

    accuracy                           0.50      1345
   macro avg       0.50      0.50      0.45      1345
weighted avg       0.50      0.50      0.45      1345
```




- Data splitting is conducted using train_test_split to partition the dataset into training and testing subsets.

- Precision, recall, F1-score, and support are provided for each class (e.g., positive and negative).
- Precision represents the proportion of true positive predictions among all positive predictions.
- Recall indicates the proportion of true positive predictions among all actual positives.
- F1-score is the harmonic mean of precision and recall, providing a balance between the two metrics.
- Support denotes the number of actual occurrences of each class in the testing set.
- The Accuracy of logistic regression was  0.5003.


## Support vector machine model

```
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report


    svc_model = SVC()

    svc_model.fit(X_train, y_train)

    y_pred = svc_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)   
```

```
Output :
Accuracy: 0.50
              precision    recall  f1-score   support

           0       0.51      0.54      0.52       680
           1       0.50      0.46      0.48       665

    accuracy                           0.50      1345
   macro avg       0.50      0.50      0.50      1345
weighted avg       0.50      0.50      0.50      1345

[[367 313]
 [356 309]]

```


- The accuracy score furnishes an overall assessment of the model's predictive performance on the testing data.
- The classification report offers detailed insights into the model's precision, recall, and F1-score for each class, facilitating a comprehensive understanding of its classification capabilities.
- Support denotes the number of actual occurrences of each class in the testing set.
- The classification report offers detailed insights into the model's precision, recall, and F1-score for each class, facilitating a comprehensive understanding of its classification capabilities.
- Accuracy of support vector machine was 0.50

## Conclusion:
1. In our prediction, we got Random Forest has highest accuracy.
2. Our accuracy was 50% only because the data was not good.
3. we are assuming, we dont have factors that are highly influencing the target variable.




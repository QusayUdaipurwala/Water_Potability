# Water_Potability
## Overview

Water is one of the most crucial resources on Earth. Life cannot survive without water. But what makes water so important? Water has several unique properties that make it highly valuable. A few such properties are listed below.

+ Water is an exceptionally good solvent - meaning it can dissolve many substances.
+ The boiling point and freezing point of water make it easily available in all three states: solid, liquid, and gaseous.
+ The specific heat of water is very high. It allows water to absorb and release heat slowly and thus regulate the temperature of the environment.
+ Being transparent, water can allow light to reach life forms submerged into it. This is vital for the survival of plant life in oceans, lakes, and rivers. Water neither contains the properties of an acid nor those of a base. Its pH level is 7, which is neutral.

This project involves the analysis and modeling of the Water Potability dataset. We will derive factors that make water safe for consumption from it. Data cleaning, preprocessing, and transformation will be carried out first to make the dataset accurate and in an ideal form for analysis. We will then carry out EDA in order to delve further into the dataset and understand the relations between different features; also, key indicators affecting water potability will be identified.

With a deep understanding of the data, a machine learning model will be designed that can predict the potability of water. Then, we evaluate the model with unseen data to check its predictive performance. If the model's initial performance is not satisfactory, it will be fine-tuned with hyperparameter tuning techniques to generate more reliable predictions.

## About Dataset
The dataset that was used in this project is from a very popular website that is known for its datasets and most of the data scientists and analyst use this site to procure data for many purposes. The website used was Kaggle.

The Dataset contains water quality metrics for 3276 different water bodies.

Dataset mainly consists of 10 attributes/columns they are:

1. **pH value**:
PH is an important parameter in evaluating the acid–base balance of water. It is also the indicator of acidic or alkaline condition of water status. WHO has recommended maximum permissible limit of pH from 6.5 to 8.5. The current investigation ranges were 6.52–6.83 which are in the range of WHO standards.

2. **Hardness:**
Hardness is mainly caused by calcium and magnesium salts. These salts are dissolved from geologic deposits through which water travels. The length of time water is in contact with hardness producing material helps determine how much hardness there is in raw water. Hardness was originally defined as the capacity of water to precipitate soap caused by Calcium and Magnesium.

3. **Solids:**
Water has the ability to dissolve a wide range of inorganic and some organic minerals or salts such as potassium, calcium, sodium, bicarbonates, chlorides, magnesium, sulfates etc. These minerals produced un-wanted taste and diluted color in appearance of water. This is the important parameter for the use of water. The water with high TDS value indicates that water is highly mineralized. Desirable limit for TDS is 500 mg/l and maximum limit is 1000 mg/l which prescribed for drinking purpose.

4. **Chloramines:**
Chlorine and chloramine are the major disinfectants used in public water systems. Chloramines are most commonly formed when ammonia is added to chlorine to treat drinking water. Chlorine levels up to 4 milligrams per liter (mg/L or 4 parts per million (ppm)) are considered safe in drinking water.

5. **Sulfate:**
Sulfates are naturally occurring substances that are found in minerals, soil, and rocks. They are present in ambient air, groundwater, plants, and food. The principal commercial use of sulfate is in the chemical industry. Sulfate concentration in seawater is about 2,700 milligrams per liter (mg/L). It ranges from 3 to 30 mg/L in most freshwater supplies, although much higher concentrations (1000 mg/L) are found in some geographic locations.

6. **Conductivity**:
Pure water is not a good conductor of electric current rather’s a good insulator. Increase in ions concentration enhances the electrical conductivity of water. Generally, the amount of dissolved solids in water determines the electrical conductivity. Electrical conductivity (EC) actually measures the ionic process of a solution that enables it to transmit current. According to WHO standards, EC value should not exceeded 400 μS/cm.

7. **Organic_carbon**:
Total Organic Carbon (TOC) in source waters comes from decaying natural organic matter (NOM) as well as synthetic sources. TOC is a measure of the total amount of carbon in organic compounds in pure water. According to US EPA < 2 mg/L as TOC in treated / drinking water, and < 4 mg/Lit in source water which is use for treatment.

8. **Trihalomethanes**:
THMs are chemicals which may be found in water treated with chlorine. The concentration of THMs in drinking water varies according to the level of organic material in the water, the amount of chlorine required to treat the water, and the temperature of the water that is being treated. THM levels up to 80 ppm is considered safe in drinking water.

9. **Turbidity**:
The turbidity of water depends on the quantity of solid matter present in the suspended state. It is a measure of light emitting properties of water and the test is used to indicate the quality of waste discharge with respect to colloidal matter. The mean turbidity value obtained for Wondo Genet Campus (0.98 NTU) is lower than the WHO recommended value of 5.00 NTU.

10. **Potability**:
Indicates if water is safe for human consumption where 1 means Potable and 0 means Not potable.


## Stage 1: Importing and Processing the Data

One of the starting steps that we are going to take is downloading and importing our dataset into our Jupyter Notebook in a format in which we can easily make changes and read our data. That can be achieved by making use of a popular library in Python known as pandas. We can easily clean, transform, and impute data according to our needs using the popular DataFrame data structure in the Pandas library.

First, after the importation of the dataset to a pandas DataFrame, we started off with some brief exploration of the structure and statistical properties of the dataset by using some built-in methods like `info()` and `describe()`. These functions provided essential information regarding the total number of rows, the data types of each attribute, and some key statistical measures such as count, mean, median, minimum and maximum values, and quartile distributions. It really helped us get a feel for the overall shape and behavior of this dataset.

We then checked for missing values using the `isna()` and `sum()` functions. The result showed that three features contained null values: ph with 491, Sulfate with a maximum count of 781, and Trihalomethanes having 162 missing entries. This was important because identifying these gaps was necessary to ensure the reliability of the data before training models.

<img width="1000" height="648" alt="image" src="https://github.com/user-attachments/assets/ec95d1bf-eb6d-406a-9e1c-c9aad9996d6c" />

Imputation techniques were performed in order to handle the missing values. Numerical imputation is typically done using strategies like Mean, Median, and Mode. Mean imputation works best when the data follows a normal distribution with a limited number of outliers. Mode is mostly applied to categorical attributes. However, median imputation will work better for the data containing skewed distributions and outliers because of its insensitivity to extreme values. Since features such as Sulfate are rather highly variable and sometimes include skewed values, median imputation was the best and most reliable option for our data. Thus, we replaced the missing values in those columns by the median of the respective attribute.

## Stage 2: Exploratory Data Analysis

In this section we are going to visualize various plots using Python libraries such as matplotlib and seaborn. Our main aim in this section is to understand our data in detail using different plots; visualization makes a big difference in understanding our data. If we just stare at the numbers and data that are present in our dataset to understand what is happening, then that is going to take a lot of time and effort. That is why we use visualization; when we visually plot our data, it becomes more clear what our data is saying and how the different attributes correlate with our target variable, which is the potability of water.

<img width="1253" height="406" alt="image" src="https://github.com/user-attachments/assets/7b5aa87b-89b8-4211-a67c-cacde233161d" />

The `describe()` function in pandas is one of the most used and the most important functions out there; it gives us a lot of information in terms of data and values. It helps us understand the distribution, spread, and potential outliers that are present in our dataset.

1. Distribution:

The mean value that is present for each and every attribute represents the central tendency of that feature. If the central tendency or the mean is much higher than the median (50% value) of the feature, then that suggests the data might be skewed. The min and max values of every feature tell us about the maximum and the minimum values that are present in the feature; they give us an idea about the spread of the data. These different statistics (`min`, `max`, `mean`, and `median`), when compared with each other, help us understand whether the data is symmetrical, skewed, or contains extreme values.

2. Spread:

The standard deviation helps us understand the spread of the data in that particular attribute. If the standard deviation is high, then the values are more spread apart from one another; if the standard deviation is lower, that suggests that the values are closer to each other and more clustered around the mean value.

The different quartiles, 25%, 50%, and 75%, tell us about how the data or values are spread across the feature. If the range between the 25% percentile and 75% percentile, basically the interquartile range, is high, then the overall variability of that feature is high, and vice versa.

3. Potential outliers:

We can also figure out the amount of potential outliers that can be present in our dataset by making use of the interquartile rule. The rule states that outliers are generally less than 25% - 1.5 x IQR and greater than 75% + 1.5 x IQR. If the maximum value of a feature is way greater than the 75% percentile, then that suggests there may be outliers present in the higher-end values of the feature, and if the minimum value is way less than the 25% percentile, then that suggests there may be outliers present in the lower-end values of the feature.

The next step that we took was to analyse our target variable which is the `Potability` attribute. We mostly wanted to check the balance between both the classes, whether the classes are balanced or imbalanced.

<img width="558" height="581" alt="image" src="https://github.com/user-attachments/assets/92825cb7-8dce-49eb-89ca-5c942c34491b" />

From the visualization we can understand that data present for not potable water meaning undrinkable water is greater than data present for drinkable water basically potable water.

### Univariate Analysis

In this step of EDA, each quantitative feature was explored separately with histograms, boxplots, and distribution plots to identify the central tendency, shape of distribution, existence of skewness, and potential presence of outliers in the data.

+ Histograms were used to show whether pH, Hardness, and Solids would follow normal or skewed distributions.
+ Boxplots were used to determine features with extremely large or variable values such as Solids and Conductivity.
+ Distribution plots showed smooth KDE curves to make the density patterns of each feature more visible. sible.

This analysis provided insight into what features might need to be transformed-scaled or normalized-and which might greatly influence the model owing to outliers.

<img width="991" height="1087" alt="image" src="https://github.com/user-attachments/assets/b84389c3-305c-4cd4-80c8-bf3690bbb4f7" />

**Histograms:** Histograms are one of the most useful plots out there. Histograms are basically used to identify the distribution of the data that is present in the data set by making use of bins and calculating the number of data points that belong in that bin, basically the frequency of data. Bins are intervals that are equal in their ranges; we make use of bins because if there are multiple values in a feature or an array/list, it becomes very clustered and not very informative. That is why we make use of bins. Histograms also are used to identify the spread of the data and even, in rare cases, to identify the outliers that are present in the data. In distributions there can be various types of distributions, such as normal, skewed, uniform, etc.

<img width="1003" height="759" alt="image" src="https://github.com/user-attachments/assets/cf46685c-0a37-4469-a169-eef0648b462f" />

Here, when we tried to create a boxplot, we noticed that the values in the feature 'Solids' have an enormous scale, which affects how the other features are visualized in the above visualization. The combined visualization of boxplot results in a compressed representation of other features in the dataset. So we then individually visualized every feature in the given dataset.

<img width="1489" height="1205" alt="image" src="https://github.com/user-attachments/assets/4d5e7f01-0708-4e72-96f4-ba62aa59f749" />

**Box Plot:** Box plots help us to quickly understand various aspects such as spread, center, outliers, and skewness of the data present in the attributes of a dataset. The rectangular box in the middle of the box plot represents 50%, basically the middle of the data, where there are Q1 and Q3 that represent the 25th and 75th percentiles of the IQR (Inter-Quartile Range). The IQR can be measured by `IQR = Q3 - Q1`. The IQR gives us information related to the spread of the data; if the IQR value is higher, then the data is widely spread (high variability); otherwise, the data is tightly packed (low variability). The middle line that is present in the box plot represents the median, or the 50% value of the data. If the median line is closer to Q1, that suggests that the data is right-skewed; if the median line is closer to Q3, that suggests that the data is left-skewed; if the median line is fairly in the middle, then it suggests that the data is fairly symmetric. The end of the whiskers on both sides represents the minimum and the maximum value present in that feature. The length of the whiskers also suggests whether the data in the feature is right-skewed or left-skewed. If the whiskers are longer in the right section, then the data is right-skewed, and if the whiskers in the left are longer, that means that the data is left-skewed. The outliers in a box plot are basically the values that are larger or lower than the maximum and the minimum value.

<img width="1489" height="1190" alt="image" src="https://github.com/user-attachments/assets/3336d814-a85c-4992-9469-3d7f1c83428d" />

**KDE Plot:** KDE plots are basically histograms but with a line that hepls us to visualize the curve that is being formed by the data from all the attributes in the dataset. This curve basically helps us to visualize the distribustion of the data to identify whether the data is normal, skewed, uniform, binomial, etc.

### Target Based Analysis

<img width="1489" height="949" alt="image" src="https://github.com/user-attachments/assets/835e74e4-dddf-439a-b560-8f42230db4a9" />

Here what we did is basically we compared all the features that were present in our dataset and compared them with our target variable, which is the potability of water. We took this step to better understand how every feature in the dataset interacts with the target variable. Here we see that there are a lot of outliers present in the dataset, though most of the data is not skewed.

### Correlation Analysis

In correlation analysis we basically check the correlation of all the features with the target attribute to understand which features are more important than others. This analysis will give us an idea of which features to focus on more and which features will make the most impact while building a predictive model.

<img width="1187" height="476" alt="image" src="https://github.com/user-attachments/assets/89c4e34a-f49c-47a0-b8e2-54111cbde05c" />

<img width="994" height="689" alt="image" src="https://github.com/user-attachments/assets/8aedcf9c-2a5f-4db8-83bf-27e1193fadca" />

Here also is a great example that shows us how visualization makes a difference. When we just used the code `df.corr()`, we could still see all the numbers, and if we took some efforts, we could have understood what the values in the dataframe were trying to depict, but when we used a heatmap to show the correlation of different features, it became a lot more clear what those values were trying to say. Basically in a heatmap the values that are darker mean that those features have a higher correlation with each other. Over here we can observe that the features apparently do not have a strong correlation with the target attribute that is potability; that was expected because it is not an easy target to predict. There are various features that affect the potability of water, and hence it is a difficult task to predict the potability of water in the environment.

### Pairplots And Scatter Plots

<img width="2308" height="2211" alt="image" src="https://github.com/user-attachments/assets/5909105b-a4b6-4e54-b814-c1838b9367a0" />

The analysis of pairplot shows that the data contains features that are fairly uncorrelated and also there is a strong overlap between potable and non-potable classes. None of the feature pairs are able to visually discriminate between classes, and most of the univariate distributions show heavy overlap in density curves. Therefore, this finally confirms that the water potability classification is a pretty complex nonlinear problem, with weak predictive power of individual features. Hence, tree-based ensemble models (Random Forest, XGBoost) and nonlinear classifiers such as SVM with RBF kernel would be better suited than linear models.

## Stage 3: Model Training and Evaluation

After understanding the structure and quality of our dataset through EDA, building predictive models for water potability will be the next step. Since our features contain skewed numerical distributions, several outliers, and a target variable that is a little imbalanced, model performance will vary significantly depending on the algorithm's assumptions and robustness.

In order to evaluate this fairly, we experiment with a set of diverse machine learning models: both linear and non-linear algorithms. We also include one model that isn't expected to work well for this dataset, aiming to evoke an understanding of how several algorithms fail whenever their assumptions are violated.

The models we are going to evaluate include:
1. Logistic Regression
2. Support Vector Machine
3. Gaussian NB
4. DecisionTreeClassifier
5. Random Forest Classifier

we will split our data into training set and testing set with a ratio of 80% to 20%. Where 80% data is used for training the model and 20% data is used to test the model then we are going to stadardize our training data set because some models perform better when the data is on the same scale such as Logistic Regression, SVM, and Gaussian NB. To answer the question of why we are splitting the data and then standardizing the data is because of data leakage. If we did the opposite where we standardized our data and then splitted the result would be that the model would learn about the testing data even before the testing were to begin this is called data leakage!.

**1. Logistic Regression:**

<img width="539" height="455" alt="image" src="https://github.com/user-attachments/assets/f6743e88-09ef-4f1e-b59b-72eedb1aecfb" />

<img width="530" height="178" alt="image" src="https://github.com/user-attachments/assets/70041c76-deeb-4d78-9eca-ec1500d462a8" />

The Logistic Regression model gave a good baseline but did not perform well on this classification task. From the confusion matrix (TN = 211, TP = 118, FP = 201, FN = 126), we can tell that the model struggles to correctly classify both classes and, in particular, class 1 examples representing potable water samples. Such a high number of false positives (201) and false negatives (126) means that the decision boundary learned by Logistic Regression fails to capture the underlying patterns in the data. This is somewhat expected because the relationships present in this dataset are very likely nonlinear and influenced by several features that interact with one another. Overall, the model indicates that simple linear classifiers are insufficient for this problem and more powerful models like Decision Trees, Random Forests, and Gradient Boosting should be tried.

**2. Support Vector Mechnine**

<img width="539" height="455" alt="image" src="https://github.com/user-attachments/assets/33e8a7fd-0052-4588-ab22-eb3f488f77c5" />

<img width="547" height="176" alt="image" src="https://github.com/user-attachments/assets/18b85a64-78b5-475d-89f5-05958e5eb2c3" />

After training the SVM model, there was a considerable improvement from the Logistic Regression model in terms of capturing the complexity of the dataset. In this case, the accuracy was 64%, performing strongly on class 0 at an F1-score of 0.71. It also performed above average for class 1 with an F1-score of 0.52. The confusion matrix results indicate that this model has correctly predicted 290 non-potable and 129 potable samples. However, it has also made a very critical mistake of predicting 122 unsafe water samples as safe, which would be considered a high false positive rate in real-world scenarios, and it also misclassified 115 safe samples as unsafe.

**3. Gaussian Naive Bayes**

<img width="539" height="455" alt="image" src="https://github.com/user-attachments/assets/1aa35abc-9fee-464b-93e2-b06855f77a11" />

<img width="538" height="187" alt="image" src="https://github.com/user-attachments/assets/cb8bccf5-2bf3-4e34-8f99-d7a10bc2dd6a" />

After training the Gaussian Naive Bayes model, we found that while it had an overall accuracy of 63%, this performance was very imbalanced between the two classes. The model performed well on the non-potable class, with high recall (0.87) and high F1-score (0.75), meaning that it is very reliable at discerning unsafe water samples. The predicted performance on the potable class was significantly weaker, with recall at 0.21 and F1-score at 0.30, which means the model has incorrectly classified a large number of safe water samples as unsafe (192 false negatives). These results point to Gaussian Naive Bayes not being the best fit for this dataset because of its assumptions of normal distribution and feature independence that are far from what is actually observed in the structure and skewness of our data. Although it can be trusted for the detection of unsafe water, it would not be a good choice when classifying potable water accurately is of interest.

**4. Decision Tree Classifier**

<img width="539" height="455" alt="image" src="https://github.com/user-attachments/assets/2edb8600-b571-4c52-83b4-68045d4ee57b" />

<img width="516" height="180" alt="image" src="https://github.com/user-attachments/assets/8fcfbfad-ff85-4b56-8be1-50dc3aa01adc" />

The Decision Tree Classifier produced an accuracy of 58%, lower compared to our previous models. The confusion matrix shows that the model performed moderately in predicting non-potable water (TN = 254), but it generated a high number of false positives (FP = 158), which means many unsafe water samples were predicted as safe. Though it captured 51% recall for potable water, the precision to this class was only 44%, meaning that there is frequent misclassification when predicting safe water. Overall, this Decision Tree struggled to generalize well and showed some signs of overfitting, which is common for unpruned trees. These results indicate that the basic Decision Tree model is not very reliable on this dataset, and its performance can get better by the application of some hyperparameter tuning-like limitations of depth or pruning techniques-or even by switching to ensemble methods such as Random Forest.

**5. Random Forest Classification**

<img width="539" height="455" alt="image" src="https://github.com/user-attachments/assets/a3152c71-a34a-4158-bc75-22ed791e9ed2" />

<img width="531" height="182" alt="image" src="https://github.com/user-attachments/assets/ffb217ff-93e4-4a6d-8981-37296de77c32" />

Training of the Random Forest Classifier provided insights into the capacity of ensemble methods in handling the Water Potability dataset. It achieved an accuracy of 67%, outperforming previous models developed to this date. This demonstrates that the complex nonlinear relationship within the data is effectively modeled by the Random Forest. Most impressively, its performance was strong on the "Not Potable" class, with a recall of 0.85 and an F1-score of 0.77, showing high capability in classifying unsafe water. However, similar to the previous models, the "Potable" class was adversely affected by class imbalance with a recall of only 0.38. Results here indicate that while the Random Forest exhibits robustness and stability, further enhancements may be necessary through techniques such as class balancing or feature engineering to improve the model's capability for the correct identification of positive drinkable water samples.

### Hyperparameter Tuning

We trained and evaluated multiple models and found out their metrics such as accuracy, precision, recall, f1-score, True Negative, False Positive, False Negative, True Positive and after analysing them we figured out the two best performing models which were SVM and Random Forest Classifier. We are now going to perform Hyperparameter Tuning to further enchance these models performance, their ability to correctly predict the potability of water.

**1. Support Vector Machine (Hyperparameter Tuning)**

Hyperparameter tuning will help us in arriving at the most suitable setting for SVM through different combinations of parameters such as C, kernel, gamma, and degree. We use GridSearchCV that automatically tests all combinations using cross-validation and picks the best performing model. Likewise, to ensure proper scaling and avoid data leakage, we make sure we go with a Pipeline.

<img width="539" height="455" alt="image" src="https://github.com/user-attachments/assets/2c3d11a4-9e45-4829-9a2d-b0ffd74024dc" />

<img width="519" height="183" alt="image" src="https://github.com/user-attachments/assets/5e818fd6-fb3e-4559-af9d-26adbaa92b02" />

After the application of hyperparameter tuning with GridSearchCV, there was an improvement in the performance of the Support Vector Machine model from its default setting. The tuned SVM reached an accuracy of 67%, with a significant boost in precision and F1-score for the potable water class (class 1). The model was highly reliable in classifying non-potable water, achieving as high as 79% recall for class 0. The prediction of potable water remains high, with the recall for class 1 at just 46%, meaning that it still missed most of the true potable cases. The tuned model gave 325 true negatives and 112 true positives, thus showing an improved balance but still biased toward the majority class. Generally speaking, the tuning improved the generalization capability of the SVM and thereby its ability to detect more potable samples compared to the untuned version, though the class imbalance problem continues to affect performance.

**2. Random Forest Regression**

<img width="539" height="455" alt="image" src="https://github.com/user-attachments/assets/70336b2d-8203-4f11-85e1-941879e836ae" />

<img width="505" height="174" alt="image" src="https://github.com/user-attachments/assets/16a47c83-4d63-40ff-a6a9-1c2b7e8d0b76" />

The Random Forest Classifier achieved an accuracy of 0.69 after hyperparameter tuning, which shows a small improvement compared to the untuned version. The classifier achieved its best results with class 0 through a recall score of 0.88 and an F1-score of 0.78. This proves the model successfully detects most of the class 0 examples. The system shows weak performance with class 1 because it achieved a recall score of 0.37 and an F1-score of 0.47. The model maintains a high number of false negatives because it produces 154 incorrect positive case predictions. The model demonstrates higher precision rates for both classes after tuning, but class 1 recall continues to be difficult because of class distribution within the dataset. Hyperparameter tuning resulted in a small performance boost, but additional methods like class balancing and feature optimization need to be implemented for better class 1 detection.

## Conclusion

+ Based on evaluations of all available machine learning models, Random Forest Classifiers with hyperparameter tuning performed best overall based on achieving the greatest accuracy of 0.69. Random Forest had a high rate of correctly classifying Non-Potable Water as indicated by the recall score of 0.88 and F1-score of 0.78. Therefore, Random Forest is reliable when identifying Non-Potable Water and could potentially serve as one of the best algorithms when predicting Non-Potable Water.

+ Conversely, when focusing on class 1, Potable Water, the tuned SVM model yielded more balanced results than the Random Forest Classifier. Although SVM's accuracy was slightly lower than that of Random Forest, SVM achieved better recall for the positive class than Random Forest and is more effective at correctly identifying Potable Water samples. This can be very significant for use cases where using false negatives could be problematic. In summary, the SVM can provide a good balance between precision and recall between both classes.

+ For all models, a difficulty in detecting class 1 as potable water resulted in a low recall for class 1 across all models. Thus, there may be an imbalance of data between classes or an overlap in the feature values of the two classes. There are also several improvements that could improve the performance of models going forward including the use of SMOTE, class weighting or additional feature engineering. In general, Random Forest and SVM were the best models, with Random Forest excelling at accuracy while SVM had the highest recall.


| Model                       | Accuracy | Precision(0) | Precision(1) | Recall(0) | Recall(1) | F1-score(0) | F1-score(1) |
|----------------------------|----------|--------------|--------------|-----------|-----------|-------------|-------------|
| Logistic Regression         | 0.65     | 0.69         | 0.54         | 0.82      | 0.36      | 0.75        | 0.43        |
| KNN Classifier             | 0.64     | 0.67         | 0.52         | 0.78      | 0.38      | 0.72        | 0.44        |
| Gaussian Naive Bayes       | 0.61     | 0.66         | 0.49         | 0.74      | 0.40      | 0.69        | 0.44        |
| Decision Tree Classifier   | 0.58     | 0.68         | 0.44         | 0.62      | 0.51      | 0.65        | 0.47        |
| Random Forest Classifier   | 0.67     | 0.70         | 0.60         | 0.85      | 0.38      | 0.77        | 0.46        |
| Tuned SVM (Best Params)    | 0.67     | 0.71         | 0.56         | 0.79      | 0.46      | 0.75        | 0.51        |
| Tuned Random Forest        | 0.69     | 0.70         | 0.65         | 0.88      | 0.37      | 0.78        | 0.47        |















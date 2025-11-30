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

Here also is a great example that shows us how visualization makes a difference. When we just used the code df.corr(), we could still see all the numbers, and if we took some efforts, we could have understood what the values in the dataframe were trying to depict, but when we used a heatmap to show the correlation of different features, it became a lot more clear what those values were trying to say. Basically in a heatmap the values that are darker mean that those features have a higher correlation with each other. Over here we can observe that the features apparently do not have a strong correlation with the target attribute that is potability; that was expected because it is not an easy target to predict. There are various features that affect the potability of water, and hence it is a difficult task to predict the potability of water in the environment.

### Pairplots And Scatter Plots

<img width="2308" height="2211" alt="image" src="https://github.com/user-attachments/assets/5909105b-a4b6-4e54-b814-c1838b9367a0" />

The analysis of pairplot shows that the data contains features that are fairly uncorrelated and also there is a strong overlap between potable and non-potable classes. None of the feature pairs are able to visually discriminate between classes, and most of the univariate distributions show heavy overlap in density curves. Therefore, this finally confirms that the water potability classification is a pretty complex nonlinear problem, with weak predictive power of individual features. Hence, tree-based ensemble models (Random Forest, XGBoost) and nonlinear classifiers such as SVM with RBF kernel would be better suited than linear models.






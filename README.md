# Water_Potability
### Overview

Water is one of the most crucial resources on Earth. Life cannot survive without water. But what makes water so important? Water has several unique properties that make it highly valuable. A few such properties are listed below.

+ Water is an exceptionally good solvent - meaning it can dissolve many substances.
+ The boiling point and freezing point of water make it easily available in all three states: solid, liquid, and gaseous.
+ The specific heat of water is very high. It allows water to absorb and release heat slowly and thus regulate the temperature of the environment.
+ Being transparent, water can allow light to reach life forms submerged into it. This is vital for the survival of plant life in oceans, lakes, and rivers. Water neither contains the properties of an acid nor those of a base. Its pH level is 7, which is neutral.

This project involves the analysis and modeling of the Water Potability dataset. We will derive factors that make water safe for consumption from it. Data cleaning, preprocessing, and transformation will be carried out first to make the dataset accurate and in an ideal form for analysis. We will then carry out EDA in order to delve further into the dataset and understand the relations between different features; also, key indicators affecting water potability will be identified.

With a deep understanding of the data, a machine learning model will be designed that can predict the potability of water. Then, we evaluate the model with unseen data to check its predictive performance. If the model's initial performance is not satisfactory, it will be fine-tuned with hyperparameter tuning techniques to generate more reliable predictions.

### About Dataset
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

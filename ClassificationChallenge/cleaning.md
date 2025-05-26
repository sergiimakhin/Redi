Below is a comprehensive overview of the tasks and steps performed to clean the dataset, as shown in the provided code. The process involves handling missing values, correcting errors, transforming features, encoding categorical variables, and scaling numerical features to prepare the dataset for a logistic regression model. The steps are organized by the type of feature (numerical, categorical, and date-related) and include explanations of the rationale behind each task.

---

### **Overview of Data Cleaning Tasks**

The dataset consists of two DataFrames: `df` (labeled data with 37,636 rows and 26 columns) and `udf` (unlabeled data with similar structure but missing the `label` column). The goal is to clean both datasets consistently to ensure they are ready for modeling, addressing issues like missing values, outliers, inconsistent formats, and high-cardinality categorical features.

#### **1. Initial Setup and Column Cleaning**
- **Task**: Remove uninformative columns and standardize column names.
- **Steps**:
  - **Drop `ID` column**: The `ID` column is removed from both `df` and `udf` as it provides no predictive value.
  - **Standardize column names**: A function `clean_column_names` is defined to remove whitespace and convert column names to lowercase. Applied to both DataFrames.
  - **Fix specific column names**: Renamed `engin_size` to `engine_size` and `runned_miles` to `mileage` for clarity.
- **Rationale**: Standardizing column names ensures consistency and avoids errors in downstream processing. Dropping uninformative columns reduces noise.

#### **2. Check for Missing Values and Duplicates**
- **Task**: Verify the dataset for missing values and duplicates.
- **Steps**:
  - **Check for NaNs**: Used `df.isnull().sum().any()` to confirm no missing values in `df`.
  - **Check for duplicates**: Used `df.duplicated().any()` to confirm no duplicate rows in `df`.
- **Rationale**: Ensuring no missing values or duplicates confirms the dataset's integrity before further processing.

#### **3. Numerical Feature Cleaning**
Numerical features include `adv_year`, `adv_month`, `adv_day`, `reg_year`, `mileage`, `price`, `seat_num`, `door_num`, `issue_id`, `repair_complexity`, `repair_cost`, and `repair_hours`. Each is cleaned and transformed as follows:

- **Advertisement Date Features (`adv_year`, `adv_month`, `adv_day`)**:
  - **Task**: Correct errors and combine into a single date feature.
  - **Steps**:
    - Identified 250 rows in `df` and 100 in `udf` with `adv_year` = 202, likely a typo for 2020. Replaced with 2020.
    - Found `adv_month` = 13 in one row of `df`, replaced with 12.
    - Combined `adv_year`, `adv_month`, and `adv_day` into `adv_date` using `pd.to_datetime`.
    - Calculated `adv_recency` as the number of days from `adv_date` to a reference date (2022-01-01).
    - Applied log transformation to `adv_recency` (`adv_recency_log`) due to skewed distribution.
    - Scaled `adv_recency_log` using `RobustScaler` to create `adv_recency_scaled`.
    - Created cyclic features for `month` and `weekday` using sine and cosine transformations to capture their cyclical nature.
    - Dropped `adv_year`, `adv_month`, `adv_day`, `adv_date`, `adv_recency`, and `adv_recency_log`.
  - **Rationale**: Combining date components into a single feature reduces dimensionality. Log transformation and scaling handle skewness and outliers, while cyclic encoding preserves the cyclical nature of months and weekdays for logistic regression.

- **Registration Year (`reg_year`)**:
  - **Task**: Convert to vehicle age and handle skewness.
  - **Steps**:
    - Calculated `vehicle_age` as 2022 minus `reg_year`.
    - Applied log transformation (`vehicle_age_log`) due to left-skewed distribution.
    - Scaled using `RobustScaler` to create `vehicle_age_scaled`.
    - Dropped `reg_year`, `vehicle_age`, and `vehicle_age_log`.
  - **Rationale**: Vehicle age is more interpretable than absolute year for modeling. Log transformation and scaling mitigate skewness and outliers.

- **Mileage (`mileage`)**:
  - **Task**: Correct negative values and handle skewness.
  - **Steps**:
    - Identified 134 negative `mileage` values in `df` and 73 in `udf`, converted to positive using `abs()`.
    - Applied log transformation (`mileage_log`) due to right-skewed distribution.
    - Scaled using `RobustScaler` to create `mileage_scaled`.
    - Dropped `mileage` and `mileage_log`.
  - **Rationale**: Negative values were likely typos. Log transformation and scaling normalize the distribution and reduce outlier impact.

- **Price (`price`)**:
  - **Task**: Handle skewness and outliers.
  - **Steps**:
    - Noted heavy right skew in `price`. Applied log transformation (`log_price`).
    - Identified 1,958 outliers in `df` and 898 in `udf` before transformation, reduced to 1,130 and 540 after log transformation.
    - Scaled `log_price` using `RobustScaler` to create `price_scaled`.
    - Dropped `price`, `log_price`, and redundant `value` (highly correlated with `price`).
  - **Rationale**: Log transformation and scaling reduce the impact of extreme values, which are common in car prices.

- **Seat Number (`seat_num`)**:
  - **Task**: Treat as categorical due to discrete nature and long tail.
  - **Steps**:
    - Binned into categories: `2-3`, `4-5`, `6-7`, `8+` based on frequency and practical significance.
    - One-hot encoded the binned feature (`seat_num_binned`), creating `seat_num_4-5`, `seat_num_6-7`, `seat_num_8+`.
    - Dropped `seat_num` and `seat_num_binned`.
  - **Rationale**: Binning reduces dimensionality and groups rare values, while one-hot encoding prepares the feature for modeling.

- **Door Number (`door_num`)**:
  - **Task**: Handle zero values and treat as categorical.
  - **Steps**:
    - Identified 606 zero-door vehicles in `df` (likely missing data), replaced with mode (5 doors).
    - Binned into `2-3`, `4-5`, `6+` categories.
    - One-hot encoded to create `door_num_4-5`, `door_num_6+`.
    - Dropped `door_num` and `door_num_binned`.
  - **Rationale**: Imputing zeros with the mode preserves data. Binning and one-hot encoding simplify the feature.

- **Issue ID (`issue_id`)**:
  - **Task**: Reduce dimensionality and treat as categorical.
  - **Steps**:
    - Binned into `0`, `1-2`, `3+` based on frequency and assumed defect count.
    - One-hot encoded to create `issue_id_1-2`, `issue_id_3+`.
    - Dropped `issue_id` and `issue_id_binned`.
  - **Rationale**: Binning simplifies the feature while preserving its ordinal nature.

- **Repair Complexity (`repair_complexity`)**:
  - **Task**: Treat as categorical and reduce dimensionality.
  - **Steps**:
    - Binned into `1`, `2`, `3-4` categories.
    - One-hot encoded to create `repair_complexity_2`, `repair_complexity_3-4`.
    - Dropped `repair_complexity` and `repair_complexity_binned`.
  - **Rationale**: Binning groups low-frequency values, and one-hot encoding prepares the feature for modeling.

- **Repair Cost (`repair_cost`)**:
  - **Task**: Handle negative values and skewness.
  - **Steps**:
    - Converted negative values to positive using `abs()`.
    - Applied log transformation (`repair_cost_log`) due to right skew.
    - Scaled using `RobustScaler` to create `repair_cost_scaled`.
    - Dropped `repair_cost` and `repair_cost_log`.
  - **Rationale**: Log transformation and scaling normalize the distribution and mitigate outliers.

- **Repair Hours (`repair_hours`)**:
  - **Task**: Handle extreme skewness and outliers.
  - **Steps**:
    - Applied log transformation (`repair_hours_log`), but distribution remained skewed.
    - Binned into `0-2`, `2-10`, `10-100`, `100-10000` categories to handle outliers.
    - One-hot encoded to create `repair_hours_2-10`, `repair_hours_10-100`, `repair_hours_100-10000`.
    - Dropped `repair_hours` and `repair_hours_log`.
  - **Rationale**: Binning is more effective than scaling for extremely skewed data, reducing outlier impact.

#### **4. Categorical Feature Cleaning**
Categorical features include `maker`, `genmodel`, `genmodel_id`, `color`, `bodytype`, `engine_size`, `gearbox`, `fuel_type`, and `issue`. These are processed to handle high cardinality and prepare for modeling.

- **Maker, Genmodel, and Genmodel ID (`maker`, `genmodel`, `genmodel_id`)**:
  - **Task**: Reduce cardinality and encode effectively.
  - **Steps**:
    - Dropped `maker` and `genmodel` as `genmodel_id` captures their information.
    - Grouped `genmodel_id` values with fewer than 376 occurrences (1% of `df`) into an `Other` category, reducing unique values from 195 to 26 in `df`.
    - Applied target encoding to `genmodel_id_grouped` using the mean of the `label` feature with additive smoothing (smoothing factor = 10) to create `genmodel_id_encoded`.
    - Dropped `genmodel_id` and `genmodel_id_grouped`.
  - **Rationale**: Target encoding reduces dimensionality while preserving predictive information. Smoothing prevents overfitting for rare categories.

- **Color (`color`)**:
  - **Task**: Handle high cardinality and rare values.
  - **Steps**:
    - Grouped colors with fewer than 376 occurrences into `Other`, reducing unique values from 22 to 13.
    - Applied target encoding to `color_grouped` with smoothing to create `color_encoded`.
    - Dropped `color` and `color_grouped`.
  - **Rationale**: Target encoding is effective for high-cardinality features, and grouping rare colors reduces noise.

- **Body Type (`bodytype`)**:
  - **Task**: Simplify and encode.
  - **Steps**:
    - Grouped body types with fewer than 376 occurrences into `Other`, reducing unique values from 15 to 9.
    - One-hot encoded `bodytype_grouped` to create columns like `bodytype_Hatchback`, `bodytype_SUV`, etc.
    - Aligned `udf` with `df` to ensure consistent one-hot encoded columns.
    - Dropped `bodytype` and `bodytype_grouped`.
  - **Rationale**: One-hot encoding is suitable for low-cardinality categorical features, and grouping rare types reduces dimensionality.

- **Engine Size (`engine_size`)**:
  - **Task**: Convert to numeric and handle outliers.
  - **Steps**:
    - Removed `L` suffix and converted to numeric (`engine_size_numeric`).
    - Replaced erroneous `999.0L` with NaN, then imputed with median.
    - Scaled using `RobustScaler` to create `engine_size_scaled`.
    - Dropped `engine_size` and `engine_size_numeric`.
  - **Rationale**: Converting to numeric allows scaling, and median imputation handles outliers without distorting the distribution.

- **Gearbox (`gearbox`)**:
  - **Task**: Simplify and encode.
  - **Steps**:
    - Combined `Hybrid` and `Semi-Automatic` into `Automatic` due to low frequency.
    - Converted to binary (`gearbox_binary`: 1 for Automatic, 0 for Manual).
    - Dropped `gearbox`.
  - **Rationale**: Binary encoding simplifies the feature for modeling.

- **Fuel Type (`fuel_type`)**:
  - **Task**: Handle high cardinality and rare values.
  - **Steps**:
    - Replaced `still_Diesel_but_you_found_an_easteregg` with `Diesel`.
    - Grouped fuel types with fewer than 376 occurrences into `Other`, reducing unique values from 11 to 3.
    - One-hot encoded to create `fuel_type_Diesel`, `fuel_type_Other`.
    - Dropped `fuel_type`.
  - **Rationale**: One-hot encoding is effective for low-cardinality features after grouping rare values.

- **Issue (`issue`)**:
  - **Task**: Clean and encode.
  - **Steps**:
    - Stripped whitespace and replaced spaces with underscores for consistency.
    - One-hot encoded to create columns like `issue_Flat_Tyres`, `issue_Transmission_Issue`, etc.
    - Dropped `issue`.
  - **Rationale**: One-hot encoding captures the distinct categories of vehicle issues.

#### **5. Date Feature Engineering**
- **Breakdown and Repair Dates (`breakdown_date`, `repair_date`)**:
  - **Task**: Create a feature for repair duration.
  - **Steps**:
    - Converted `breakdown_date` and `repair_date` to datetime.
    - Calculated `days_to_repair` as the difference in days.
    - Applied log transformation (`days_to_repair_log`) due to right skew.
    - Created a binary feature `is_same_day_repair` (1 if `days_to_repair_log` = 0, else 0).
    - Scaled `days_to_repair_log` using `RobustScaler` to create `days_to_repair_log_scaled`.
    - Dropped `breakdown_date`, `repair_date`, and `days_to_repair_log`.
  - **Rationale**: Repair duration is a meaningful feature, and the binary indicator captures same-day repairs. Log transformation and scaling handle skewness.

#### **6. Final Checks and Alignment**
- **Task**: Ensure consistent features and data types.
- **Steps**:
  - Converted boolean columns to integers for consistency.
  - Aligned `udf` columns with `df` (excluding `label`) using `reindex`.
  - Checked for multicollinearity by computing correlations, identifying strong correlations (e.g., `engine_size_scaled` and `repair_complexity_3-4` at 0.706512).
- **Rationale**: Consistent data types and column alignment ensure compatibility for modeling. Correlation analysis helps identify potential multicollinearity issues.

#### **7. Final Dataset**
- **Result**:
  - `df` shape: (37,636, 52) with 51 numeric features (36 binary, 15 continuous) and 1 target (`label`).
  - `udf` shape: (16,130, 51) with the same 51 features.
  - All features are numeric, scaled, or encoded, ready for logistic regression.

---

### **Key Considerations**
- **Outlier Handling**: Used `RobustScaler` for numerical features and binning for features with extreme outliers (e.g., `repair_hours`) to reduce their impact.
- **Categorical Encoding**: Used target encoding for high-cardinality features (`genmodel_id`, `color`) and one-hot encoding for low-cardinality features (`bodytype`, `fuel_type`, etc.) to balance dimensionality and information retention.
- **Feature Engineering**: Created meaningful features like `vehicle_age`, `adv_recency`, and `days_to_repair` to capture temporal relationships.
- **Consistency**: Ensured `udf` aligns with `df` to prevent errors during model inference.

This cleaning process prepares the dataset for logistic regression by addressing skewness, outliers, and high cardinality while preserving predictive information.
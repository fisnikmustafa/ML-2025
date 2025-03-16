# Machine Learning - 2025

## University and Contributors
![Universiteti-i-Prishtinës-Hasan-Prishtina (1)](https://github.com/user-attachments/assets/8f672579-f493-4d33-9352-6fc645d38733)
- **University**: University of Pristina
- **Faculty**: Faculty of Electrical and Computer Engineering
- **Mentor**: Dr. Sc. Mërgim H. HOTI
- **Students**:
  - Altin MUSLIU
  - Edi MORINA
  - Fisnik MUSTAFA

## Table of Contents
- [Dataset](#dataset)
- [Overview](#overview)
- [Goal](#goal)
- [Phase1: Preparing the Model](#phase)

## Dataset

This script works with two datasets:

[Workout & Fitness Tracker Dataset](https://www.kaggle.com/datasets/adilshamim8/workout-and-fitness-tracker-data): This dataset contains fitness-related attributes such as Age, Height (cm), Weight (kg), Resting Heart Rate (bpm), BMI, and other workout statistics. It serves as the primary dataset, and most of the features come from this file.

[FitLife: Health & Fitness Tracking Dataset](https://www.kaggle.com/datasets/jijagallery/fitlife-health-and-fitness-tracking-dataset): This dataset includes additional health-related information, such as Blood Pressure, Stress Level, Smoking Status, Health Condition, etc. These attributes are merged with the first dataset based on matching features (like Age, Height (cm), Weight (kg), and BMI), allowing us to enhance the information in the first dataset and make it more unique.

The primary focus is on matching the individuals between the two datasets using common features, while adding supplementary information from the second dataset to enrich the data for further analysis.

## Overview

This repository focuses on developing a machine learning model to generate personalized fitness workout plans. The project leverages a dataset of fitness and workout data to create a tailored experience based on individual needs and goals. The ultimate aim is to help users optimize their workouts by recommending exercises, intensity, and schedules that align with their fitness levels and objectives.

## Goal

The goal of this project is to build an intelligent system that can analyze user fitness data and suggest effective workout plans. By applying machine learning techniques, we aim to make fitness recommendations that are adaptive, efficient, and tailored to the specific needs of individuals.

## Phase 1: Preparing the Model

The first phase of this project focuses on preparing the dataset and building the foundational machine learning model.


### 1. Data Loading & Cleaning

The script starts by loading two datasets, performs the following:
- **BMI Calculation**: Adds a new column `BMI` to `dataset1` using the formula `Weight / Height^2`.
- **Gender Standardization**: Maps gender labels (`Male`, `Female`, `Other`) to a consistent format (`M`, `F`, `Other`).
- **Renaming Columns**: Ensures column names are consistent between the two datasets.

### 2. Feature Matching Using KD-Tree

To match individuals in `dataset1` and `dataset2` based on the specified features (e.g., Age, Height, Weight, Resting Heart Rate, BMI):
- Data is normalized using `MinMaxScaler`.
- A KD-Tree is created for efficient nearest-neighbor search.
- For each gender (`M`, `F`), the script matches individuals from `dataset1` to the closest individuals in `dataset2` based on the weighted features.
- For those labeled as "Other," the script matches based on features without considering gender.

### 3. Data Merging & Outlier Removal

- After matching the data, the script merges relevant columns from `dataset2` with `dataset1` based on the nearest neighbor results.
- Outliers are detected and removed using the IQR method for `BMI`, `Blood Pressure Systolic`, and `Blood Pressure Diastolic`.

### 4. Encoding & Feature Engineering

- **Label Encoding**: Columns with categorical values such as Mood and Smoking Status are encoded using `LabelEncoder`.
- **One-Hot Encoding**: Categorical columns such as Workout Type and Intensity are one-hot encoded using `pd.get_dummies()`.

### 5. Data Visualizations

- **Boxplots**: The script generates boxplots for numeric features such as `BMI`, `Blood Pressure Systolic`, and `Blood Pressure Diastolic` to visualize the distribution of values and identify any remaining outliers.

### 6. Data Scaling

The numeric features are standardized using `StandardScaler` to ensure they have zero mean and unit variance, which is essential for machine learning models.

### 7. Data Splitting

The final dataset is split into training and testing sets (80%/20%) using `train_test_split()`.

### 8. Saving the Final Dataset

The cleaned and preprocessed dataset is saved as `merged_dataset_filter_by_gender3.csv` in the current working directory.

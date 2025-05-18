import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Phase 1: Load and Preprocess Data
df = pd.read_csv('data/nutrition_physical-activity_obesity.csv')

# Filter for obesity data (QuestionID == Q036)
df_obesity = df[df['QuestionID'] == 'Q036'].copy()

# Create binary target: 1 if Data_Value >= 30, 0 otherwise
df_obesity['High_Obesity'] = np.where(df_obesity['Data_Value'] >= 30, 1, 0)

# Handle missing Data_Value (marked with '~')
df_obesity = df_obesity[df_obesity['Data_Value'] != '~']
df_obesity['Data_Value'] = df_obesity['Data_Value'].astype(float)

# Phase 1: Outlier Handling for Data_Value
plt.figure(figsize=(8, 6))
sns.boxplot(y=df_obesity['Data_Value'], color='#1f77b4')
plt.title('Phase 1: Box Plot of Obesity Rates (Q036)')
plt.ylabel('Obesity Rate (%)')
plt.savefig('phase1_obesity_boxplot.png')

Q1 = df_obesity['Data_Value'].quantile(0.25)
Q3 = df_obesity['Data_Value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_obesity['Data_Value'] = df_obesity['Data_Value'].clip(lower=lower_bound, upper=upper_bound)
print(f"Outliers capped: Data_Value clipped to [{lower_bound:.2f}, {upper_bound:.2f}]")


# Phase 1 Visualization 1: Obesity Rate Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_obesity['Data_Value'], bins=30, kde=True, color='#1f77b4')
plt.axvline(x=30, color='red', linestyle='--', label='Threshold (30%)')
plt.title('Phase 1: Distribution of Obesity Rates (Q036)')
plt.xlabel('Obesity Rate (%)')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('phase1_obesity_distribution.png')


# Feature Engineering
features = ['YearStart', 'LocationAbbr', 'Age(years)', 'Sex', 'Education', 'Income', 'Race/Ethnicity']

# Check available physical activity data
pa_questions = ['Q043', 'Q044']
df_pa = df[df['QuestionID'].isin(pa_questions)][['YearStart', 'LocationAbbr', 'Age(years)', 'Sex',
                                                'Education', 'Income', 'Race/Ethnicity',
                                                'QuestionID', 'Data_Value']]

# Fill missing values in df_pa
for col in ['Age(years)', 'Sex', 'Education', 'Income', 'Race/Ethnicity']:
    df_pa[col] = df_pa[col].fillna('Unknown')
df_pa['Data_Value'] = pd.to_numeric(df_pa['Data_Value'], errors='coerce')

# Debug: Check physical activity data
print("Physical Activity Data (Q043, Q044) Rows:")
print(df_pa.head())
print("Unique QuestionIDs in df_pa:", df_pa['QuestionID'].unique())
print("Non-null Data_Value counts in df_pa:")
print(df_pa.groupby('QuestionID')['Data_Value'].count())

# Pivot with simplified index to maximize data retention
pivot_index = ['YearStart', 'LocationAbbr']
df_pa_pivot = df_pa.pivot_table(index=pivot_index, columns='QuestionID', values='Data_Value',
                                aggfunc='first').reset_index()

# Debug: Check columns and non-null counts
print("Columns in df_pa_pivot:", df_pa_pivot.columns.tolist())
print("Non-null counts in df_pa_pivot:")
print(df_pa_pivot[['Q043', 'Q044']].notnull().sum())

# Phase 1: Outlier Handling for Q043, Q044
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_pa_pivot[['Q043', 'Q044']], palette=['#1f77b4', '#ff7f0e'])
plt.title('Phase 1: Box Plot of Physical Activity Data (Q043, Q044)')
plt.ylabel('Percentage (%)')
plt.savefig('phase1_pa_boxplot.png')
plt.show()

for col in ['Q043', 'Q044']:
    if col in df_pa_pivot.columns:
        Q1 = df_pa_pivot[col].quantile(0.25)
        Q3 = df_pa_pivot[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_pa_pivot[col] = df_pa_pivot[col].clip(lower=lower_bound, upper=upper_bound)
        print(f"Outliers capped: {col} clipped to [{lower_bound:.2f}, {upper_bound:.2f}]")

# Phase 1 Visualization 2: Physical Activity Data Availability
counts = {
    'Raw Data': df_pa.groupby('QuestionID')['Data_Value'].count(),
    'Pivoted Data': df_pa_pivot[['Q043', 'Q044']].notnull().sum(),
    'Merged Data': df_obesity.merge(df_pa_pivot, on=['YearStart', 'LocationAbbr'], how='left')[['Q043', 'Q044']].notnull().sum()
}
counts_df = pd.DataFrame(counts, index=['Q043', 'Q044']).T
plt.figure(figsize=(10, 6))
counts_df.plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
plt.title('Phase 1: Physical Activity Data Availability (Q043, Q044)')
plt.xlabel('Data Stage')
plt.ylabel('Non-Null Count')
plt.xticks(rotation=0)
plt.legend(title='QuestionID')
plt.tight_layout()
plt.savefig('phase1_pa_availability.png')


# Phase 1 Visualization 3: Feature Distribution (YearStart and Categorical Features)
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.boxplot(y=df_obesity['YearStart'], color='#1f77b4')
plt.title('Phase 1: YearStart Distribution')
plt.subplot(2, 2, 2)
sns.countplot(y=df_obesity['LocationAbbr'], order=df_obesity['LocationAbbr'].value_counts().index[:10], color='#ff7f0e')
plt.title('Phase 1: Top 10 Locations')
plt.subplot(2, 2, 3)
sns.countplot(y=df_obesity['Race/Ethnicity'], order=df_obesity['Race/Ethnicity'].value_counts().index[:5], color='#2ca02c')
plt.title('Phase 1: Top 5 Race/Ethnicity')
plt.tight_layout()
plt.savefig('phase1_feature_distribution.png')


# Convert pivoted Data_Value to float
for col in pa_questions:
    if col in df_pa_pivot.columns:
        df_pa_pivot[col] = pd.to_numeric(df_pa_pivot[col], errors='coerce')
    else:
        print(f"Warning: Column {col} not found in df_pa_pivot. Adding as zero-filled.")
        df_pa_pivot[col] = 0

# Merge physical activity data
df_merged = df_obesity.merge(df_pa_pivot, on=['YearStart', 'LocationAbbr'], how='left')

# Debug: Check merge results
print("Non-null counts in df_merged for Q043, Q044:")
print(df_merged[['Q043', 'Q044']].notnull().sum())

# Phase 1 Visualization 4: Correlation Heatmap (Numerical Features)
numerical_cols = ['YearStart', 'Q043', 'Q044']
corr_matrix = df_merged[numerical_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', vmin=-1, vmax=1)
plt.title('Phase 1: Correlation Heatmap of Numerical Features')
plt.savefig('phase1_correlation_heatmap.png')


# Encode categorical variables
le = LabelEncoder()
categorical_cols = ['LocationAbbr', 'Age(years)', 'Sex', 'Education', 'Income', 'Race/Ethnicity']
for col in categorical_cols:
    df_merged[col] = df_merged[col].fillna('Unknown')
    df_merged[col] = le.fit_transform(df_merged[col])

# Impute missing values with median by LocationAbbr
for col in pa_questions:
    if col in df_merged.columns:
        df_merged[col] = df_merged.groupby('LocationAbbr')[col].transform(lambda x: x.fillna(x.median()))
        df_merged[col] = df_merged[col].fillna(df_merged[col].median())
    else:
        df_merged[col] = 0

# Define feature set and target
feature_cols = features + [col for col in pa_questions if col in df_merged.columns]
X = df_merged[feature_cols]
y = df_merged['High_Obesity']

# Phase 2: Train Model with Hyperparameter Tuning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', None]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Phase 2 Visualization 1: Grid Search F1-Scores
results = pd.DataFrame(grid_search.cv_results_)
pivot_table = results[(results['param_min_samples_split'] == 5) & (results['param_class_weight'].isna())].pivot(
    index='param_max_depth',
    columns='param_n_estimators',
    values='mean_test_score'
)
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='Blues', cbar_kws={'label': 'F1-Score'})
plt.title('Phase 2: Grid Search F1-Scores (min_samples_split=5, class_weight=None)')
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.savefig('phase2_grid_search.png')


# Phase 2 Visualization 2: Parameter Impact
results_subset = results[results['param_class_weight'].isna()]
plt.figure(figsize=(10, 6))
for depth in [10, 20]:
    subset = results_subset[results_subset['param_max_depth'] == depth]
    plt.plot(subset['param_n_estimators'], subset['mean_test_score'], marker='o', label=f'max_depth={depth}')
plt.title('Phase 2: F1-Score vs. n_estimators by max_depth')
plt.xlabel('n_estimators')
plt.ylabel('F1-Score')
plt.legend()
plt.grid(True)
plt.savefig('phase2_parameter_impact.png')


# Phase 2 Visualization 3: Training Data Balance
plt.figure(figsize=(8, 6))
sns.countplot(x=y_train, palette=['#1f77b4', '#ff7f0e'])
plt.title('Phase 2: Training Data Class Distribution')
plt.xlabel('High_Obesity (0: Low, 1: High)')
plt.ylabel('Count')
plt.savefig('phase2_class_balance.png')


# Phase 3: Evaluate Model
y_pred = best_rf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}, Precision: {precision_score(y_test, y_pred):.2f}, Recall: {recall_score(y_test, y_pred):.2f}, F1: {f1_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Phase 3 Visualization 1: Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, color='#1f77b4')
plt.title('Phase 3: Feature Importance for Obesity Classification')
plt.savefig('phase3_feature_importance.png')


# Phase 3 Visualization 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low Obesity', 'High Obesity'],
            yticklabels=['Low Obesity', 'High Obesity'])
plt.title('Phase 3: Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('phase3_confusion_matrix.png')


# Phase 3 Visualization 3: ROC Curve
y_prob = best_rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Phase 3: ROC Curve for Obesity Classification')
plt.legend(loc='lower right')
plt.savefig('phase3_roc_curve.png')


# Phase 3 Visualization 4: Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='#1f77b4', lw=2, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Phase 3: Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.savefig('phase3_precision_recall.png')


# Phase 3 Visualization 5: Prediction Distribution
plt.figure(figsize=(10, 6))
sns.histplot(y_prob, bins=30, kde=True, color='#1f77b4', label='Predicted Probabilities (Class 1)')
plt.title('Phase 3: Distribution of Predicted Probabilities for High Obesity')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('phase3_prediction_distribution.png')

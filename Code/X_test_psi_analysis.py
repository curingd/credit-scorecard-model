# %%
# Import packages
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Paths for training and holdout datasets
REPO_ROOT = Path(__file__).resolve().parent.parent    
X_TRAIN_CSV = REPO_ROOT / "Datasets" / "X_train.csv"
X_TEST_PSI_CSV = REPO_ROOT / "Datasets" / "X_test_psi.csv"

# %%
# Import X_test_psi to analyse distribution
psi_df = pd.read_csv(X_TEST_PSI_CSV)
train_df = pd.read_csv(X_TRAIN_CSV)

# %%
# Combined datasets for plotting
psi_df['dataset'] = 'X_test_psi'
train_df['dataset'] = 'X_train'
combined_df = pd.concat([train_df, psi_df], ignore_index=True)

# %%
# Analyse distributions of revenue
plt.figure(figsize=(8, 6))
revenue_dist = sns.histplot(data=combined_df, x='revenue', stat='density', element='poly', hue='dataset', common_norm=False)

plt.title('Distributions of Revenues')
plt.xlabel('Revenue ($)')
plt.ylabel('Frequency Density')
plt.xlim((0, 300000))
plt.show()

# %%
# Analyse distributions of debt-to-income ratio
plt.figure(figsize=(8, 6))
dti_dist = sns.histplot(data=combined_df, x='dti_n', stat='density', element='poly', hue='dataset', common_norm=False)

plt.title('Distributions of DTI Ratio')
plt.xlabel('DTI Ratio')
plt.ylabel('Frequency Density')
plt.xlim((0, 50))
plt.show()

# %%
# Analyse distributions of loan amount
plt.figure(figsize=(8, 6))
loan_amount_dist = sns.histplot(data=combined_df, x='loan_amnt', stat='density', element='poly', hue='dataset', common_norm=False)

plt.title('Distributions of Loan Amounts')
plt.xlabel('Loan Amount ($)')
plt.ylabel('Frequency Density')
plt.show()

# %%
# Analyse distributions of FICO score
plt.figure(figsize=(8, 6))
loan_amount_dist = sns.histplot(data=combined_df, x='fico_n', stat='density', element='poly', hue='dataset', common_norm=False)

plt.title('Distributions of FICO Scores')
plt.xlabel('FICO Score')
plt.ylabel('Frequency Density')
plt.xlim((650, 850))
plt.show()

# %%
# Analyse distributions of unemployment rate
plt.figure(figsize=(8, 6))
loan_amount_dist = sns.histplot(data=combined_df, x='unemployment_rate', stat='density', element='poly', hue='dataset', common_norm=False)

plt.title('Distributions of Unemployment Rate')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Frequency Density')
plt.show()

# %%
# Analyse distributions of home ownership type
plt.figure(figsize=(8, 6))
loan_amount_dist = sns.histplot(data=combined_df, x='home_ownership_n', stat='density', multiple='dodge', hue='dataset', shrink=0.8, common_norm=False)

plt.title('Distributions of Home Ownership Type')
plt.xlabel('Home Ownership Type')
plt.ylabel('Frequency Density')
plt.show()

# %%
# Analyse distributions of home ownership type
plt.figure(figsize=(8, 6))
loan_amount_dist = sns.histplot(data=combined_df, x='purpose', stat='density', multiple='dodge', hue='dataset', shrink=0.8, common_norm=False)

plt.title('Distributions of Loan Purpose')
plt.xlabel('Loan Purpose')
plt.ylabel('Frequency Density')
plt.xticks(rotation=90)
plt.show()

# %%
train_zip_props = combined_df.loc[combined_df['dataset'] == 'X_train', 'zip_code'].value_counts(normalize=True)
psi_zip_props = combined_df.loc[combined_df['dataset'] == 'X_test_psi', 'zip_code'].value_counts(normalize=True)
zip_props = pd.DataFrame({'X_train': train_zip_props, 'X_test_psi': psi_zip_props}).fillna(0)
different_zip_codes = zip_props.index[np.abs(zip_props['X_train'] - zip_props['X_test_psi']) > 0.0005]


# Analyse distributions of zip code
plt.figure(figsize=(12, 6))
loan_amount_dist = sns.histplot(data=combined_df[combined_df['zip_code'].isin(different_zip_codes)].sort_values(by='zip_code'), x='zip_code', stat='density', multiple='dodge', hue='dataset', shrink=0.8, common_norm=False)

plt.title('Distributions of Zip Codes with Largest Proportional Differences')
plt.xlabel('Zip Code')
plt.ylabel('Frequency Density')
plt.xticks(rotation=90)
plt.show()



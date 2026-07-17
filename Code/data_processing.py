# %%
# Import packages
from pathlib import Path
import numpy as np
import pandas as pd
import datetime as dt

# %%
# Paths for unemployment rate and credit spread datasets
REPO_ROOT = Path(__file__).resolve().parent.parent    
UNRATE_CSV = REPO_ROOT / "Datasets" / "alfred_unrate.csv"
CREDIT_CSV = REPO_ROOT / "Datasets" / "alfred_moodys_baa_10y_spread.csv"
LOANS_CSV = REPO_ROOT / "Datasets" / "LC_loans.csv"
OUT = Path(__file__).resolve().parent

# %%
# Snapshot of unemployment rate dataset
employment_df = pd.read_csv(UNRATE_CSV)
employment_df.head()

# %%
# Data types
employment_df.info()

# %%
# Are the UNRATE_20260605 and UNRATE_20260702 columns ever different?
filter = (employment_df['UNRATE_20260605'] == employment_df['UNRATE_20260702'])
employment_df[filter].count()

# %%
# Return rows for which UNRATE_20260605 and UNRATE_20260702 are different
not_filter = filter = (employment_df['UNRATE_20260605'] != employment_df['UNRATE_20260702'])
employment_df[not_filter]

# %%
# Clean dataset
employment_df['observation_date'] = pd.to_datetime(employment_df['observation_date'])
employment_df['year'] = employment_df['observation_date'].dt.year
employment_df['month'] = employment_df['observation_date'].dt.month
employment_df.drop(columns=['observation_date', 'UNRATE_20260702'], inplace=True)
employment_df.dropna(inplace=True)
employment_df.rename(columns={'UNRATE_20260605':'unemployment_rate'}, inplace=True)
employment_df.head()

# %%
# Lag unemployment rate by one month to compensate for lagged data reporting
num_rows = employment_df.shape[0]
employment_df['unemployment_rate'] = employment_df['unemployment_rate'].shift(1)
employment_df.drop([0], inplace=True)
employment_df.reset_index(drop=True, inplace=True)
employment_df.head()

# %%
# Import credit spreads
credit_spread_df = pd.read_csv(CREDIT_CSV)
credit_spread_df.head()

# %%
# Clean dataset
credit_spread_df['observation_date'] = pd.to_datetime(credit_spread_df['observation_date'])
credit_spread_df['year'] = credit_spread_df['observation_date'].dt.year
credit_spread_df['month'] = credit_spread_df['observation_date'].dt.month
credit_spread_df['day'] = credit_spread_df['observation_date'].dt.day
credit_spread_df.drop(columns=['observation_date'], inplace=True)
credit_spread_df.rename(columns={'BAA10Y':'credit_spread'}, inplace=True)
credit_spread_df.head()

# %%
# Replace credit spreads with monthly averages
credit_spread_df = credit_spread_df.groupby(by=['year','month']).mean()
credit_spread_df.drop(columns=['day'], inplace=True)
credit_spread_df.head()


# %%
# Prepare LC_loans for joining with employment and credit data
LC_loans_df = pd.read_csv(LOANS_CSV)
LC_loans_df['issue_d'] = pd.to_datetime(LC_loans_df['issue_d'], format='%b-%Y')
LC_loans_df['year'] = LC_loans_df['issue_d'].dt.year
LC_loans_df['month'] = LC_loans_df['issue_d'].dt.month
LC_loans_df.drop(columns=['issue_d'], inplace=True)
LC_loans_df.head()

# %%
# Joining LC_loans with employment and credit data
LC_loans_df = LC_loans_df.merge(employment_df, on=['year', 'month'], how='left')
LC_loans_df = LC_loans_df.merge(credit_spread_df, on=['year', 'month'], how='left')
LC_loans_df.head()

# %%
# Restore original issue_d column as datetime
LC_loans_df['issue_d'] = pd.to_datetime(
    LC_loans_df[['year', 'month']].assign(day=1)
).dt.strftime('%b-%Y')
LC_loans_df.drop(columns=['year', 'month'], inplace=True)
LC_loans_df.head()

# %%
# Check dataset for nulls in columns
LC_loans_df.info()

# %%
# Change NaN values in zip_code, title and desc to "Unknown" for model building
LC_loans_df.fillna(value='Unknown', inplace=True)

# %%
# Write output CSV
LC_loans_df.to_csv(f'{OUT}/LC_loans_unrate_credit.csv', index=False)



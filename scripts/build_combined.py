import pandas as pd

X_clinical = pd.read_csv('data/processed/X_clinical.csv')
X_genomic = pd.read_csv('data/processed/X_genomic.csv')
y = pd.read_csv('data/processed/y.csv')

# merge on sampleId
X_combined = pd.merge(X_clinical, X_genomic, on='sampleId', how='inner')

# sanity checks
assert len(X_combined) == 455, f"Expected 455 rows, got {len(X_combined)}"
assert X_combined['sampleId'].nunique() == len(X_combined), "Duplicate sampleIds after merge"
assert X_combined.isnull().sum().sum() == 0, "NaNs present after merge"

print(f'X_combined shape: {X_combined.shape}')
print(f'Clinical features: {X_clinical.shape[1] - 2}')   # minus patientId and sampleId
print(f'Genomic features: {X_genomic.shape[1] - 1}')     # minus sampleId
print(f'Total features: {X_combined.shape[1] - 2}')               # minus patientId and sampleId
print(f'y shape: {y.shape}')
print(f'\nFeature columns:\n{X_combined.columns.tolist()}')

X_combined.to_csv('data/processed/X_combined.csv', index=False)
print('\nSaved: data/processed/X_combined.csv')
import pandas as pd

# loading inputs
mutations = pd.read_csv('data/raw/mutations.csv')
X_genomic_0 = pd.read_csv('data/processed/X_genomic_0.csv')

# dropping genes with low signal AND no strong LUAD-specific rationale
GENES_TO_DROP = [
    'CCND1', 'CDK4', 'IDH1', 'IDH2', 'MDM2',
    'MYC', 'VHL', 'FGFR3', 'ESR1', 'AR',
    'CDK6', 'FGFR2', 'MYCN'
]
mutations = mutations[~mutations['gene_symbol'].isin(GENES_TO_DROP)]


# filter to somatic mutations only (dropped 25 UNKNOWN status mutations)
mutations = mutations[mutations['mutationStatus'] == 'SOMATIC']

mutation_matrix = (
    mutations
    .groupby(['sampleId', 'gene_symbol'])
    .size()         # count mutations per sample-gene pair
    .gt(0)          # convert to binary
    .astype(int)    
    .unstack(fill_value=0)  # genes become columns, missing -> 0
    .reset_index()
)
mutation_matrix.columns.name = None # clean up column index name

# align to X_genomic_0 sample index given by Jason
# keep only sample that survived
# samples with no panel mutations get all zeros
target_samples = X_genomic_0['sampleId']

mutation_matrix = (
    target_samples
    .to_frame()
    .merge(mutation_matrix, on='sampleId', how='left')
    .fillna(0)
)

# results
n_with_mutations = mutation_matrix.iloc[:, 1:].any(axis=1).sum()
n_all_zero = len(mutation_matrix) - n_with_mutations
print(f'Samples with >= 1 panel mutation: {n_with_mutations}')
print(f'Samples with no panel mutations: {n_all_zero}')


# concatenate with X_genomic_0
X_genomic = pd.concat(
    [mutation_matrix, X_genomic_0.drop(columns='sampleId')],
    axis=1
)

# sanity checks
assert X_genomic['sampleId'].nunique() == len(X_genomic), "Duplicate sampleIds detected"
assert X_genomic.isnull().sum().sum() == 0, "NaNs present in final matrix"
print(f'\nFinal X_genomic shape: {X_genomic.shape}')
print(f'Columns: sampleId + {mutation_matrix.shape[1]-1} mutation features + {X_genomic_0.shape[1]-1} summary features')


# save file
X_genomic.to_csv('data/processed/X_genomic.csv', index=False)
print('Saved: data/processed/X_genomic.csv')


# checking the per-gene mutation frequency
gene_cols = [c for c in X_genomic.columns if c not in ['sampleId'] and c in mutations['gene_symbol'].unique()]
freq = X_genomic[gene_cols].mean().sort_values(ascending=False)
print('\nPer-gene mutation frequency (% of samples):')
print((freq * 100).round(1).to_string())
low_signal = freq[freq < 0.05].index.tolist()
if low_signal:
    print(f'\nGenes mutated in <5% of samples (low signal, consider dropping): {low_signal}')
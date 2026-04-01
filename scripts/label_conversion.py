import pandas as pd

y = pd.read_csv('data/processed/y.csv')

# priority ordering will be defined as:
# CNS -> Bone -> Liver -> Adrenal -> Lung -> Pleura -> LN -> Non-metastatic
PRIORITY = [
    'EVER_MET_SITE_CNS',
    'EVER_MET_SITE_BONE',
    'EVER_MET_SITE_LIVER_BILIARY_TRACT',
    'EVER_MET_SITE_ADRENAL',
    'EVER_MET_SITE_LUNG',
    'EVER_MET_SITE_PLEURA',
    'EVER_MET_SITE_LN',
]

LABEL_MAP = {
    'EVER_MET_SITE_CNS': 'CNS',
    'EVER_MET_SITE_BONE': 'Bone',
    'EVER_MET_SITE_LIVER_BILIARY_TRACT': 'Liver',
    'EVER_MET_SITE_ADRENAL': 'Adrenal',
    'EVER_MET_SITE_LUNG': 'Lung',
    'EVER_MET_SITE_PLEURA': 'Pleura',
    'EVER_MET_SITE_LN': 'LN',
}

# assign a single label per patient
def assign_label(row):
    for col in PRIORITY:
        if row[col] == 1:
            return LABEL_MAP[col]
    return 'Non-Metastatic'

y_multiclass = y.copy()
y_multiclass['label'] = y.apply(assign_label, axis=1)


# sanity checks and checking class distribution
print('Class distribution:')
print(y_multiclass['label'].value_counts())
print(f'\nTotal samples: {len(y_multiclass)}')
assert y_multiclass['label'].isnull().sum() == 0, "Null labels detected"

# save files
y_multiclass[['label']].to_csv('data/processed/y_multiclass.csv', index=False)
print('\n Saved: data/processed/y_multiclass.csv')
import pandas as pd
import matplotlib.pyplot as plt
from download_data import get_clinical_patient_data, get_clinical_sample_data

STUDY_ID = "luad_mskcc_2023_met_organotropism"
clinical_patient= get_clinical_patient_data(STUDY_ID)
clinical_sample = get_clinical_sample_data(STUDY_ID)

clinical_sample['patientId'] = clinical_sample['sampleId'].str.extract(r'(P-\d+)') # get the patient id from sample id
df = pd.merge(clinical_patient, clinical_sample, how='inner', on='patientId')
df = df[df['GROUP_NO'].isin(['Group1', 'Group2'])]
percent_missing_per_feature = df.isnull().mean().sort_values(ascending=False)
drop_cols = percent_missing_per_feature[percent_missing_per_feature > .2].index.to_list()
df = df.drop(columns=drop_cols)

# drop rows with missing values
df = df.dropna(axis=0)
# drop rows where race is unknown
df_clean = df[df['RACE'] != 'Unknown']


target_cols = ['EVER_MET_SITE_ADRENAL', 
               'EVER_MET_SITE_BONE', 
               'EVER_MET_SITE_CNS',
               'EVER_MET_SITE_LIVER_BILIARY_TRACT', 
               'EVER_MET_SITE_LN',
               'EVER_MET_SITE_LUNG', 
               'EVER_MET_SITE_PLEURA']


# remove metadata other than patientId, sampleId, and GENE_PANEL
metadata_cols = [
    'GROUP_NO',
    'CANCER_TYPE', 'CANCER_TYPE_DETAILED', 'ONCOTREE_CODE',
    'PRIMARY_SITE', 'SAMPLE_CLASS', 'SAMPLE_COVERAGE',
    'SAMPLE_COUNT', 'INSTITUTE', 'IN_MATCHED',
    'IMPACT_PRIMARY_GROUP', 'SOMATIC_STATUS'
]

leak_cols = [
    'ADRENAL_STATUS', 'BONE_STATUS', 'CNS_STATUS',
    'LIVER_STATUS', 'LN_STATUS', 'LUNG_STATUS', 'PLEURA_STATUS',
    # Future outcome info
    'DEATH', 'OS_MONTHS', 'OS_STATUS', 'FU_2YRS', 'METASTATIC_BURDEN',
    # Post sample treatment
    'POST_SAMPLE_CHEMOTHERAPY', 'POST_SAMPLE_IMMUNOTHERAPY',
    'POST_SAMPLE_TARGETED', 'POST_SAMPLE_TX', 'POST_SAMPLE_XRT',
    # Adjuvant (post surgery)
    'ADJUVANT', 'ADJUVANT_CHEMOTHERAPY', 'ADJUVANT_IMMUNOTHERAPY',
    'ADJUVANT_TARGETED', 'ADJUVANT_XRT', 'ADJUVANT_THERAPY',
]

# keep only neoadjuvant feature, which is yes if any of the following are true, and no if all are false
# exclude pathways as well, this is encoded in the mutation matrix
multicollinearity = ['NEOADJUVANT_CHEMOTHERAPY', 'NEOADJUVANT_IMMUNOTHERAPY',
    'NEOADJUVANT_TARGETED', 'NEOADJUVANT_XRT', 'HIPPO', 'MYC_PATH', 'NOTCH','NRF2', 'PI3K', 'RTK_RAS', 'TGF_BETA', 'TP53_PATH', 'WNT']

genomic_cols = [
    'FGA', 'FRACTION_GENOME_ALTERED', 'IS_WGD',
    'MSI_SCORE', 'MSI_TYPE', 'MUTATION_COUNT',
    'PLOIDY', 'PURITY', 'TMB_NONSYNONYMOUS',
    'CELL_CYCLE']

# Drop all columns that are metadata, can cause potential leakage, and exclude genomic features as well. Separate the target from the features
y = df_clean[['patientId', 'sampleId'] + target_cols]
X = df_clean.drop(columns=target_cols + metadata_cols + leak_cols + multicollinearity + genomic_cols)

# Drop any column with one unique value, does not contribute to model
X = X.loc[:, X.nunique() > 1] 

# encode features using one hot encoding
one_hot = ['CIGARETTE_HX', 'NEOADJUVANT', 'RACE', 'SEX',
       'PREDOM_HISTO_SUBTYPE', 'PRE_SAMPLE_CHEMOTHERAPY',
       'PRE_SAMPLE_IMMUNOTHERAPY', 'PRE_SAMPLE_TARGETED', 'PRE_SAMPLE_TX',
       'PRE_SAMPLE_XRT']
X_ENCODED= pd.get_dummies(X, columns=one_hot, drop_first=True)

# clinical stage and pathological stage's ordinal values may have clinical meaning. 
# We could go one hot encoding as well,
# but we choose to preserve the ordinal information
X_ENCODED['CSTAGE'] = X_ENCODED['CSTAGE'].map({'Stage I': 1, 'Stage II': 2, 'Stage III': 3, 'Stage IV': 4})
X_ENCODED['PSTAGE'] = X_ENCODED['PSTAGE'].map({'Stage I': 1, 'Stage II': 2, 'Stage III': 3})

X_genomic = df_clean[genomic_cols]
X_genomic_enc= pd.get_dummies(X_genomic, columns=['CELL_CYCLE', 'IS_WGD', 'MSI_TYPE'], drop_first=True)

# check that X and y are of the same length
assert(len(X_genomic_enc) == len(y))
assert (len(X_ENCODED) == len(y))

#save to csv
y.to_csv('data/processed/y.csv', index=False)
X_ENCODED.to_csv('data/processed/X_clinical.csv', index=False)
X_genomic_enc.insert(0, 'sampleId', df_clean['sampleId'].values)
X_genomic_enc.to_csv('data/processed/X_genomic_0.csv', index=False)

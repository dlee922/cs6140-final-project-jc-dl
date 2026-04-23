import pandas as pd
X_combined = pd.read_csv("../data/processed/X_combined.csv")
ids = ['patientId', 'sampleId', 'GENE_PANEL']
X_combined = X_combined.drop(columns=ids)

# define interactions
X_combined['TP53_x_EGFR'] = X_combined['TP53'] * X_combined['EGFR']
X_combined['KEAP1_x_STK11'] = X_combined['KEAP1'] * X_combined['STK11']
X_combined['KEAP1_x_KRAS'] = X_combined['KEAP1'] * X_combined['KRAS']
X_combined['KRAS_x_STK11'] = X_combined['KRAS'] * X_combined['STK11']
X_combined['FGA_x_TP53'] = X_combined['FGA'] * X_combined['TP53']
X_combined['FGA_x_ERBB2'] = X_combined['FGA'] * X_combined['ERBB2']
X_combined['smoking_x_KRAS'] = (1 - X_combined['CIGARETTE_HX_Never']) * X_combined['KRAS']
X_combined['smoking_x_SMARCA4'] = (1 - X_combined['CIGARETTE_HX_Never']) * X_combined['SMARCA4']
X_combined['smoking_x_TP53'] = (1 - X_combined['CIGARETTE_HX_Never']) * X_combined['TP53']
X_combined['PSTAGE_x_SMARCA4'] = X_combined['PSTAGE'] * X_combined['SMARCA4']
X_combined['PSTAGE_x_TP53'] = X_combined['PSTAGE'] * X_combined['TP53']
X_combined['PSTAGE_x_neoadjuvant'] = X_combined['PSTAGE'] * X_combined['NEOADJUVANT_Yes']

X_combined.to_csv('data/processed/X_combined_interact.csv')

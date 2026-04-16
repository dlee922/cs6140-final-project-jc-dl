"""
Download MSK LUAD Met Organotropism data from cBioPortal
Uses their public API to get mutation and clinical data
"""
import requests
import pandas as pd
import os
import yaml
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.display import print_header

# Create data directories if they do not exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# cBioPortal API base URL
BASE_URL = "https://www.cbioportal.org/api"

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

STUDY_ID = config['study']['study_id']
MOLECULAR_PROFILE_ID = f"{STUDY_ID}_mutations"
SAMPLE_LIST_ID = f"{STUDY_ID}_sequenced"

# Gene panel — broad set of LUAD-relevant genes
# Cast a wide net here and narrow during processing
GENE_PANEL = [
    "EGFR", "KRAS", "STK11", "KEAP1", "TP53", "BRAF", "MET", "RB1",
    "CDKN2A", "NF1", "RET", "ALK", "ROS1", "ERBB2", "PIK3CA", "PTEN",
    "SMAD4", "ARID1A", "SETD2", "U2AF1", "RBM10", "ATM", "BRCA1", "BRCA2",
    "FGFR1", "FGFR2", "FGFR3", "NTRK1", "NTRK2", "NTRK3", "MAP2K1",
    "SMARCA4", "CTNNB1", "APC", "NOTCH1", "NOTCH2", "TERT", "MDM2",
    "CDK4", "CDK6", "CCND1", "MYC", "MYCN", "AR", "ESR1", "MTOR",
    "TSC1", "TSC2", "VHL", "IDH1", "IDH2"
]

print_header(f"MSK {config['study']['cancer_abbreviation']} Data Download from cBioPortal")


def get_entrez_ids(gene_symbols):
    """
    Convert Hugo gene symbols to Entrez IDs using cBioPortal API.
    The mutations endpoint requires Entrez IDs.

    Args:
        gene_symbols (list): List of Hugo gene symbols e.g. ['EGFR', 'KRAS']

    Returns:
        dict: Mapping of {hugo_symbol: entrez_id}
    """
    print(f"\n[1/4] Fetching Entrez IDs for {len(gene_symbols)} genes...")

    url = f"{BASE_URL}/genes/fetch"

    response = requests.post(
        url,
        json=gene_symbols,
        params={"geneIdType": "HUGO_GENE_SYMBOL"},
        headers={"Content-Type": "application/json"}
    )

    if response.status_code != 200:
        print(f"   Error fetching gene IDs: {response.status_code}")
        print(f"   Response: {response.text[:500]}")
        return {}

    genes = response.json()
    gene_map = {g['hugoGeneSymbol']: g['entrezGeneId'] for g in genes}

    found = len(gene_map)
    missing = [g for g in gene_symbols if g not in gene_map]
    print(f"   ✓ Found Entrez IDs for {found}/{len(gene_symbols)} genes")
    if missing:
        print(f"   Genes not found: {missing}")

    return gene_map


def get_mutations(molecular_profile_id, sample_list_id, entrez_ids):
    """
    Download mutations using sampleListId + entrezGeneIds.

    Args:
        molecular_profile_id (str): e.g. 'luad_mskcc_2023_met_organotropism_mutations'
        sample_list_id (str): e.g. 'luad_mskcc_2023_met_organotropism_sequenced'
        entrez_ids (list): List of integer Entrez gene IDs

    Returns:
        pd.DataFrame: Raw mutations data
    """
    print(f"\n[2/4] Downloading mutations...")

    url = f"{BASE_URL}/molecular-profiles/{molecular_profile_id}/mutations/fetch"

    payload = {
        "entrezGeneIds": entrez_ids,
        "sampleListId": sample_list_id
    }

    try:
        response = requests.post(
            url,
            json=payload,
            params={"projection": "DETAILED"},
            headers={"Content-Type": "application/json"},
            timeout=120
        )

        if response.status_code != 200:
            print(f"   Error: {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return pd.DataFrame()

        data = response.json()
        df = pd.DataFrame(data)

        if df.empty:
            print("   No mutations returned")
            return df

        # Extract gene symbol from nested gene dict
        if 'gene' in df.columns and isinstance(df['gene'].iloc[0], dict):
            df['gene_symbol'] = df['gene'].apply(
                lambda x: x.get('hugoGeneSymbol', '') if isinstance(x, dict) else x
            )

        print(f"   ✓ Downloaded {len(df):,} mutations")
        print(f"   ✓ Unique samples with mutations: {df['sampleId'].nunique():,}")
        print(f"   ✓ Unique genes: {df['gene_symbol'].nunique():,}")

        return df

    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def get_clinical_sample_data(study_id):
    """
    Download sample-level clinical data.
    Includes metastatic site columns (EVER_MET_SITE_*) and GROUP_NO.

    Args:
        study_id (str): cBioPortal study ID

    Returns:
        pd.DataFrame: Wide-format clinical data, one row per sample
    """
    print(f"\n[3/4] Downloading sample-level clinical data...")

    url = f"{BASE_URL}/studies/{study_id}/clinical-data"
    params = {
        "clinicalDataType": "SAMPLE",
        "projection": "DETAILED"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"   Error: {response.status_code}")
        return pd.DataFrame()

    clinical = response.json()
    df = pd.DataFrame(clinical)

    df_wide = df.pivot_table(
        index='sampleId',
        columns='clinicalAttributeId',
        values='value',
        aggfunc='first'
    ).reset_index()

    print(f"   ✓ {df_wide.shape[0]} samples × {df_wide.shape[1]} columns")

    return df_wide


def get_clinical_patient_data(study_id):
    """
    Download patient-level clinical data.
    Includes demographics: RACE, SEX, and metastatic outcome fields.

    Args:
        study_id (str): cBioPortal study ID

    Returns:
        pd.DataFrame: Wide-format patient clinical data
    """
    print(f"\n[4/4] Downloading patient-level clinical data...")

    url = f"{BASE_URL}/studies/{study_id}/clinical-data"
    params = {
        "clinicalDataType": "PATIENT",
        "projection": "DETAILED"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"   Error: {response.status_code}")
        return pd.DataFrame()

    clinical = response.json()
    df = pd.DataFrame(clinical)

    df_wide = df.pivot_table(
        index='patientId',
        columns='clinicalAttributeId',
        values='value',
        aggfunc='first'
    ).reset_index()

    print(f"   ✓ {df_wide.shape[0]} patients × {df_wide.shape[1]} columns")

    return df_wide


# Main execution
if __name__ == "__main__":
    try:
        # Step 1: Get Entrez IDs for our gene panel
        gene_map = get_entrez_ids(GENE_PANEL)

        if not gene_map:
            print("\n✗ Could not fetch gene IDs. Exiting.")
            exit(1)

        entrez_ids = list(gene_map.values())

        # Step 2: Download mutations
        mutations_df = get_mutations(MOLECULAR_PROFILE_ID, SAMPLE_LIST_ID, entrez_ids)

        if mutations_df.empty:
            print("\n✗ No mutations downloaded. Check the error messages above.")
            exit(1)

        # Step 3: Download sample-level clinical data
        clinical_sample_df = get_clinical_sample_data(STUDY_ID)

        # Step 4: Download patient-level clinical data
        clinical_patient_df = get_clinical_patient_data(STUDY_ID)

        # Save raw data
        print("\n" + "="*70)
        print("Saving raw data...")
        print("="*70)

        mutations_df.to_csv('data/raw/mutations.csv', index=False)
        print(f"✓ Saved: mutations.csv ({len(mutations_df):,} rows)")

        clinical_sample_df.to_csv('data/raw/clinical_sample.csv', index=False)
        print(f"✓ Saved: clinical_sample.csv "
              f"({clinical_sample_df.shape[0]} samples × {clinical_sample_df.shape[1]} cols)")

        clinical_patient_df.to_csv('data/raw/clinical_patient.csv', index=False)
        print(f"✓ Saved: clinical_patient.csv "
              f"({clinical_patient_df.shape[0]} patients × {clinical_patient_df.shape[1]} cols)")

        pd.DataFrame(
            list(gene_map.items()),
            columns=['hugo_symbol', 'entrez_id']
        ).to_csv('data/raw/gene_map.csv', index=False)
        print(f"✓ Saved: gene_map.csv ({len(gene_map)} genes)")

        print("\n" + "="*70)
        print("Summary")
        print("="*70)
        print(f"Mutations downloaded:  {len(mutations_df):,} variants across {len(entrez_ids)} genes")
        print(f"Samples (clinical):    {clinical_sample_df.shape[0]:,}")
        print(f"Patients (clinical):   {clinical_patient_df.shape[0]:,}")
        print(f"\n✓ Download complete. Next: run scripts/02_process_data.py")

    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()

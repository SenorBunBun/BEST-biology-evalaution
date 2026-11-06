"""
Script to process OSD-402, OSD-403, OSD-404, OSD-405 datasets.
Each dataset is read from .h5 files, metadata is extracted from the corresponding
metadata folders, and saved as individual .h5ad files with the following columns:
- Specimen ID (formerly "batch") - extracted from cell barcodes
- Strain
- Sex
- Age at Launch (in weeks)
- Duration (in days)
- Flight (1 for Space Flight, 0 for Ground Control)
- dataset (OSD ID)
"""

import os
import re
import pandas as pd
import scanpy as sc
import numpy as np

# Configuration for each dataset
DATASETS = {
    '402': {
        'h5_file': 'osd_402.h5',
        'metadata_folder': 'OSD-402_metadata_OSD-402-ISA',
        'sample_file': 's_OSD-402.txt',
        'assay_file': 'a_OSD-402_transcription-profiling_single-cell-rna-sequencing.txt',
        'tissue': 'Femur BM'  # Bone marrow of femur
    },
    '403': {
        'h5_file': 'osd_403.h5',
        'metadata_folder': 'OSD-403_metadata_OSD-403-ISA',
        'sample_file': 's_OSD-403.txt',
        'assay_file': 'a_OSD-403_transcription-profiling_single-cell-rna-sequencing.txt',
        'tissue': 'Humerus BM'  # Bone marrow of humerus
    },
    '404': {
        'h5_file': 'osd_404.h5',
        'metadata_folder': 'OSD-404_metadata_OSD-404-ISA',
        'sample_file': 's_OSD-404.txt',
        'assay_file': 'a_OSD-404_transcription-profiling_single-cell-rna-sequencing.txt',
        'tissue': 'PBMC'  # peripheral blood mononuclear cell
    },
    '405': {
        'h5_file': 'osd_405.h5',
        'metadata_folder': 'OSD-405_metadata_OSD-405-ISA',
        'sample_file': 's_OSD-405.txt',
        'assay_file': 'a_OSD-405_transcription-profiling_single-cell-rna-sequencing.txt',
        'tissue': 'Spleen'
    }
}


def extract_specimen_id(obs_name):
    """
    Extract the specimen ID from cell barcode.
    Cell barcodes typically end with -1, -2, etc.
    Returns the number after the last dash.
    """
    match = re.search(r'-(\d+)$', obs_name)
    if match:
        return int(match.group(1))
    return None


def read_metadata_txt(file_path):
    """Read tab-delimited metadata from txt file"""
    try:
        df = pd.read_csv(file_path, sep='\t')
        return df
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return pd.DataFrame()


def parse_age(age_value, unit='week'):
    """Parse age value to weeks"""
    if pd.isna(age_value):
        return 0
    try:
        return int(age_value)
    except (ValueError, TypeError):
        return 0


def parse_duration(duration_value, unit='day'):
    """Parse duration value to days"""
    if pd.isna(duration_value):
        return 0
    try:
        return int(duration_value)
    except (ValueError, TypeError):
        return 0


def parse_flight(flight_value):
    """Parse flight status to string: 'Flight' or 'Ground'"""
    if pd.isna(flight_value):
        return ''
    return 'Flight' if flight_value == 'Space Flight' else 'Ground'


def get_samples_from_assay(assay_path):
    """Get list of sample names that have single-cell RNA sequencing data"""
    if os.path.exists(assay_path):
        assay_df = read_metadata_txt(assay_path)
        if not assay_df.empty and 'Sample Name' in assay_df.columns:
            return assay_df['Sample Name'].tolist()
    return []


def create_sample_metadata_dict(sample_df, valid_samples=None):
    """
    Create a dictionary mapping sample names to their metadata.

    The sample metadata files have columns:
    - Sample Name
    - Characteristics[Strain]
    - Characteristics[Sex]
    - Factor Value[Age] + Unit (for age at launch)
    - Factor Value[Spaceflight] (for flight status)
    - Parameter Value[Duration] + Unit (for duration)
    """
    metadata_dict = {}

    for idx, row in sample_df.iterrows():
        sample_name = row.get('Sample Name', '')

        # Skip samples not in valid_samples list if provided
        if valid_samples and sample_name not in valid_samples:
            continue

        # Extract metadata
        strain = row.get('Characteristics[Strain]', '')
        sex = row.get('Characteristics[Sex]', '')

        # Age at launch (in weeks for these datasets)
        age_value = row.get('Factor Value[Age]', 0)
        age_at_launch = parse_age(age_value)

        # Duration (in days)
        duration_value = row.get('Parameter Value[Duration]', 0)
        duration = parse_duration(duration_value)

        # Flight status
        flight_value = row.get('Factor Value[Spaceflight]', '')
        flight = parse_flight(flight_value)

        metadata_dict[sample_name] = {
            'Strain': strain,
            'Sex': sex,
            'Age at Launch': age_at_launch,
            'Duration': duration,
            'Flight': flight
        }

    return metadata_dict


def process_dataset(osd_id, config, base_path):
    """
    Process a single OSD dataset.

    Mapping approach (same as data.py):
    - Row index in s_OSD-*.txt (0-indexed) + 1 = barcode suffix
    - So row 0 in metadata -> cells with barcode suffix -1
    - Row 1 in metadata -> cells with barcode suffix -2, etc.

    Args:
        osd_id: The OSD dataset ID (e.g., '402')
        config: Dictionary with paths and configuration
        base_path: Base directory path

    Returns:
        Processed AnnData object
    """
    print(f"\n{'='*60}")
    print(f"Processing OSD-{osd_id}")
    print(f"{'='*60}")

    # Paths
    h5_path = os.path.join(base_path, config['h5_file'])
    metadata_folder = os.path.join(base_path, config['metadata_folder'])
    sample_path = os.path.join(metadata_folder, config['sample_file'])

    # Read h5 file
    print(f"Reading {config['h5_file']}...")
    try:
        adata = sc.read_10x_h5(h5_path)
        print(f"  Shape: {adata.shape}")
    except Exception as e:
        print(f"  ERROR reading h5 file: {e}")
        return None

    # Make var names unique
    adata.var_names_make_unique()

    # Extract specimen IDs from cell barcodes
    # The barcode suffix (-1, -2, etc.) indicates which sample the cell belongs to
    print("Extracting Specimen IDs from cell barcodes...")
    adata.obs['Specimen_ID'] = adata.obs_names.map(extract_specimen_id)
    adata.obs['Specimen_ID'] = adata.obs['Specimen_ID'].astype('Int64')  # Nullable integer

    # Get unique specimen IDs
    unique_specimens = sorted(adata.obs['Specimen_ID'].dropna().unique())
    print(f"  Found {len(unique_specimens)} unique specimens: {list(unique_specimens)}")

    # Read sample metadata - this is the source of truth for mapping
    # Row order in this file determines the batch/specimen ID mapping
    print(f"Reading sample metadata from {config['sample_file']}...")
    sample_df = read_metadata_txt(sample_path)

    if sample_df.empty:
        print("  ERROR: Could not read sample metadata")
        return None

    print(f"  Found {len(sample_df)} samples in metadata file")

    # Initialize metadata columns with proper types
    adata.obs['Strain'] = ''
    adata.obs['Sex'] = ''
    adata.obs['Age at Launch'] = pd.array([0] * adata.n_obs, dtype='Int64')  # Nullable integer
    adata.obs['Duration'] = pd.array([0] * adata.n_obs, dtype='Int64')  # Nullable integer
    adata.obs['Flight'] = ''

    # Following the same approach as data.py:
    # Row index (0-based) in sample_df + 1 = batch/specimen ID
    print(f"\n  Mapping samples (row_index + 1 = specimen ID):")

    cells_matched = 0
    cells_unmatched = 0

    for idx, row in sample_df.iterrows():
        batch = idx + 1  # batch/specimen ID is 1-indexed
        sample_name = row.get('Sample Name', f'Sample_{batch}')

        # Check if this batch exists in the data
        if batch not in unique_specimens:
            print(f"    Row {idx} (Specimen {batch}): {sample_name} - NO CELLS IN DATA")
            continue

        mask = adata.obs['Specimen_ID'] == batch
        num_cells = mask.sum()

        # Extract metadata
        strain = row.get('Characteristics[Strain]', '')
        sex = row.get('Characteristics[Sex]', '')

        # Age at launch (in weeks for these datasets)
        age_value = row.get('Factor Value[Age]', 0)
        age_at_launch = parse_age(age_value)

        # Duration (in days)
        duration_value = row.get('Parameter Value[Duration]', 0)
        duration = parse_duration(duration_value)

        # Flight status
        flight_value = row.get('Factor Value[Spaceflight]', '')
        flight = parse_flight(flight_value)

        # Assign to cells
        adata.obs.loc[mask, 'Strain'] = strain
        adata.obs.loc[mask, 'Sex'] = sex
        adata.obs.loc[mask, 'Age at Launch'] = age_at_launch
        adata.obs.loc[mask, 'Duration'] = duration
        adata.obs.loc[mask, 'Flight'] = flight

        cells_matched += num_cells
        print(f"    Row {idx} (Specimen {batch}): {sample_name} ({num_cells} cells, {flight}, age={age_at_launch})")

    # Check for unmatched specimens
    for specimen_id in unique_specimens:
        batch = int(specimen_id)
        if batch > len(sample_df):
            mask = adata.obs['Specimen_ID'] == batch
            cells_unmatched += mask.sum()
            print(f"    WARNING: Specimen {batch} has no metadata row!")

    print(f"\n  Cells with metadata: {cells_matched}")
    print(f"  Cells without metadata: {cells_unmatched}")

    # Summary
    print(f"\n  Final dataset shape: {adata.shape}")
    print(f"  Metadata columns: {list(adata.obs.columns)}")
    print(f"\n  Flight distribution:")
    print(adata.obs['Flight'].value_counts())
    print(f"\n  Age at Launch distribution:")
    print(adata.obs['Age at Launch'].value_counts())
    print(f"\n  Strain distribution:")
    print(adata.obs['Strain'].value_counts())

    return adata


def main():
    """Main function to process all datasets"""
    # Get the directory where this script is located
    base_path = os.path.dirname(os.path.abspath(__file__))

    print("="*60)
    print("OSD Dataset Preprocessing Pipeline")
    print("="*60)
    print(f"Base path: {base_path}")

    # Process each dataset
    for osd_id, config in DATASETS.items():
        adata = process_dataset(osd_id, config, base_path)

        if adata is not None:
            # Save as h5ad
            output_path = os.path.join(base_path, f'osd_{osd_id}.h5ad')
            print(f"\n  Saving to {output_path}...")
            adata.write(output_path)
            print(f"  Saved successfully!")

    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == '__main__':
    main()

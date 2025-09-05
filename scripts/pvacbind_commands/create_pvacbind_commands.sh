#!/bin/bash

WORKING_BASE=/storage1/fs1/mgriffit/Active/immune/pvactools_percentiles
RESULTS_BASE=$WORKING_BASE/pvacbind_results
SCRATCH_BASE=/scratch1/fs1/mgriffit/pvacbind_results
FASTA_BASE=$WORKING_BASE/peptide-sequence-synthesis/data/1M_Peptides
ALLELES_FILE=$WORKING_BASE/pvacbind_valid_classI_alleles.txt

# Read alleles file into array
mapfile -t HLA_ALLELES < "$ALLELES_FILE"

# Define lengths array
LENGTHS=(8 9 10 11)

# Loop through arrays
for HLA in "${HLA_ALLELES[@]}"; do
  for LEN in "${LENGTHS[@]}"; do
    HLA_NAME="${HLA//\*/_}"
    HLA_NAME="${HLA_NAME//\:/_}"
    SCRATCH_OUTDIR=${SCRATCH_BASE}/${LEN}/${HLA_NAME}
    FINAL_OUTDIR=${RESULTS_BASE}/${LEN}/${HLA_NAME}
    RUN_NAME=${LEN}_${HLA_NAME}_AllClassI
    echo -e "\nProcessing predictions for $RUN_NAME"
    echo mkdir -p $SCRATCH_OUTDIR
    echo pvacbind run $FASTA_BASE/reference_${LEN}mer_1M_mutated_1x.fasta $RUN_NAME $HLA all_class_i $SCRATCH_OUTDIR --class-i-epitope-length $LEN --n-threads 8 --iedb-install-directory /opt/iedb --fasta-size 10000
    echo mkdir -p $FINAL_OUTDIR
    echo "cp -r $SCRATCH_OUTDIR/* $FINAL_OUTDIR"
  done
done


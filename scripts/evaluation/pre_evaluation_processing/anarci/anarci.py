"""
anarci.py — CDR-H3 Extraction Pipeline (Shell Commands Reference)
=================================================================

This file documents the three-stage shell pipeline used to extract CDR-H3
sequences from model-generated VH (heavy chain variable domain) sequences,
using the ANARCI antibody numbering tool with the Chothia scheme.

PURPOSE
-------
After the model generates paired VH:VL FASTA entries, this pipeline:
  1. Strips the VL portion (everything after the ":" delimiter) to isolate VH.
  2. Numbers the VH sequences using ANARCI with the Chothia scheme.
  3. Extracts CDR-H3 residues (Chothia positions H95–H102) into both TSV and
     FASTA formats for downstream structural evaluation.

INPUT
-----
  model_output/fasta/incremental/4813.fasta
    Paired FASTA file where each sequence line has the form "VH:VL".

OUTPUTS
-------
  generated_vh.fasta              — VH-only FASTA (VL stripped)
  generated_anarci_chothia.txt    — Raw ANARCI positional numbering output
  generated_h3_chothia.tsv        — Tab-separated: sequence_id <TAB> CDR-H3_string
  generated_h3_chothia.fasta      — FASTA of CDR-H3 sequences only

WHY CHOTHIA?
------------
Chothia is a canonical antibody numbering scheme that defines CDR loop
boundaries by structural alignment rather than sequence identity alone. It
places CDR-H3 at positions 95–102 (with alphabetic insertion codes for loops
longer than 8 residues). Using a fixed numbering scheme makes CDR-H3
boundaries comparable across diverse sequences.

HOW TO RUN
----------
  Copy and execute the three command blocks below in the shell, in order.
  Adjust the hardcoded paths if the output directory changes.

NOTE: This file is intentionally stored as a .py file to keep it alongside
the Python pipeline scripts, but the contents are bash/awk commands.
"""

"""
awk '
/^>/ {print; next}
{
  split($0,a,":");
  print a[1]
}
' /home/alanwu/Documents/iggen_model/model_output/fasta/incremental/4813.fasta >  /home/alanwu/Documents/iggen_model/model_output/anarci_files/incremental/4813/generated_vh.fasta

ANARCI -i /home/alanwu/Documents/iggen_model/model_output/anarci_files/incremental/4813/generated_vh.fasta -s chothia > /home/alanwu/Documents/iggen_model/model_output/anarci_files/incremental/4813/generated_anarci_chothia.txt

awk '
BEGIN {id=""; h3=""}

# record header is exactly "# <id>"
/^# / && $2!="ANARCI" && $2!="Domain" && $2!="Most" && $2!~ /^\|/ && $2!="Scheme" {
  if (id!="") print id "\t" h3
  id=$2
  h3=""
  next
}

# numbering lines
$1=="H" {
  pos=$2+0
  aa=$NF
  # Chothia CDR-H3 is typically H95-H102 (inclusive), insertions like 95A are included automatically
  if (pos>=95 && pos<=102 && aa!="-" && aa!=".") h3=h3 aa
  next
}

END {
  if (id!="") print id "\t" h3
}
' /home/alanwu/Documents/iggen_model/model_output/anarci_files/incremental/4813/generated_anarci_chothia.txt > /home/alanwu/Documents/iggen_model/model_output/anarci_files/incremental/4813/generated_h3_chothia.tsv

awk '
BEGIN {id=""; h3=""}
/^# / && $2!="ANARCI" && $2!="Domain" && $2!="Most" && $2!~ /^\|/ && $2!="Scheme" {
  if (id!="" && h3!="") {print ">"id; print h3}
  id=$2; h3=""; next
}
$1=="H" {
  pos=$2+0; aa=$NF
  if (pos>=95 && pos<=102 && aa!="-" && aa!=".") h3=h3 aa
  next
}
END { if (id!="" && h3!="") {print ">"id; print h3} }
      ' /home/alanwu/Documents/iggen_model/model_output/anarci_files/incremental/4813/generated_anarci_chothia.txt > /home/alanwu/Documents/iggen_model/model_output/anarci_files/incremental/4813/generated_h3_chothia.fasta
"""

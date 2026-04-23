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
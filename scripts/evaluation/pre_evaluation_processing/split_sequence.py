"""
split_sequence.py — VH/VL Chain Splitter for p-IgGen Raw Output
================================================================

PURPOSE
-------
p-IgGen (and similar antibody language models) paired sequence generation mode did not work locally. This
script recovers the boundary and produces structured output for downstream
tools (AlphaFold/ColabFold structural prediction, ANARCI numbering, etc.).

INPUT
-----
  model_output/raw_sequences.txt
    Plain text file; one concatenated VH+VL sequence per line, no header.

OUTPUTS
-------
  pairs_split.txt   — One "VH, VL" pair per line (human-readable).
  failed_lines.txt  — Lines for which no reliable split point was found.
  fasta/incremental/4813.fasta
    Paired FASTA in ColabFold multimer format: each entry has a single
    sequence line "VH:VL". ColabFold interprets ":" as a chain break,
    letting it fold VH and VL as a heterodimer without a linker.

SPLITTING STRATEGY (two-tier)
------------------------------
1. VL-start motif detection (preferred):
     Scan for known light-chain N-terminal consensus sequences (e.g.
     "DIQMTQ" for kappa chains, "QSVLTQ" for lambda chains). Search only
     past position 80 to avoid spurious matches within VH. VL starts are
     more conserved than VH ends and are therefore more reliable anchors.

2. Heavy-end motif detection (fallback):
     If no VL start is found, scan for conserved J-gene framework 4
     sequences at the VH C-terminus (e.g. "WGQGTLVTVSS"). Split
     immediately after the last (rightmost) occurrence to handle any
     repeated sub-strings appearing earlier in the sequence.

Both strategies enforce minimum length guards (VH ≥ 90 aa, VL ≥ 80 aa) to
reject implausible splits caused by partial motif matches.

HOW TO RUN
----------
  python split_sequence.py
  (No command-line arguments; edit the path constants below to change paths.)

DEPENDENCIES
------------
  Python ≥ 3.8 standard library only (pathlib).
"""

# split_and_convert_piggen_raw_to_fasta.py
#
# 1) Reads p-IgGen "raw" output where each line is a single concatenated Fv (VH+VL, no delimiter)
# 2) Splits into VH and VL using:
#    - preferred: detection of VL-start motifs (more robust)
#    - fallback: heavy-end motifs
# 3) Writes:
#    - pairs_split.txt   (VH, VL) one pair per line
#    - failed_lines.txt  lines that could not be split
#    - pairs_split.fasta paired FASTA entries with "VH:VL" (ColabFold/AlphaFold-friendly)
#
# Paths are set to your locations.

from pathlib import Path

INFILE = Path("/home/alanwu/Documents/iggen_model/model_output/raw_sequences.txt")
PAIRS_TXT = Path("/home/alanwu/Documents/iggen_model/model_output/pairs_split.txt")
FAILED_TXT = Path("/home/alanwu/Documents/iggen_model/model_output/failed_lines.txt")
PAIRS_FASTA = Path("/home/alanwu/Documents/iggen_model/model_output/fasta/incremental/4813.fasta")

# Light-chain starts are more consistent than heavy-chain ends
VL_START_MOTIFS = [
    "DIQMTQ", "DIVMTQ", "DVVMTQ",   # kappa VL (IGKV) group 1/2
    "EIVLTQ", "EIVMTQ",               # kappa VL group 3
    "QSVLTQ", "QSALTQ", "QITLK", "QVTLR",  # lambda VL starters
    "ENVLTQ", "SYELTQ", "SSELTQ", "AIQLTQ",
    "MTQSP", "VLTQSP", "LTQSP",       # partial-motif fallbacks for truncated starts
]

# Fallback heavy-end motifs (broad)
HEAVY_END_MOTIFS = [
    # These are the J-gene framework 4 (FR4) consensus sequences at the VH C-terminus.
    # Multiple variants exist due to IGHJ gene diversity and somatic mutation.
    "WGQGTLVTVSS", "WGQGTTVTVSS", "WGQGSLVTVSS", "WGQGILVTVSS", "WGQGTSVTVSS",
    "WGRGTLVTVSS", "WGHGTTVTVSS", "WGKGTTVTVSS", "WGHGTLVTVSS",
    "WGQGTLVTVSA", "WGQGTTVTVSA", "WGRGTLVTVSA", "WGKGTTVTVSA",
]

MIN_VH_LEN = 90
MIN_VL_LEN = 80


def split_by_vl_start(s: str):
    """
    Find earliest plausible VL start AFTER a safe offset to avoid matching inside VH.
    """
    # VH sequences are always >80 aa, so any motif hit before position 80 is
    # a false positive inside the heavy chain — skip it.
    safe_offset = 80
    candidates = []
    for motif in VL_START_MOTIFS:
        idx = s.find(motif, safe_offset)
        if idx != -1:
            candidates.append(idx)

    if not candidates:
        return None

    # Take the earliest hit to maximise VH length; later hits would shorten VH
    # and lengthen VL, both of which push toward implausible lengths.
    cut = min(candidates)
    vh, vl = s[:cut], s[cut:]

    if len(vh) < MIN_VH_LEN or len(vl) < MIN_VL_LEN:
        return None
    return vh, vl


def split_by_heavy_end(s: str):
    """
    Split after the LAST occurrence of a heavy-end motif.
    """
    best_cut = None
    for motif in HEAVY_END_MOTIFS:
        idx = s.rfind(motif)  # rfind avoids false positives from VH internal repeats
        if idx != -1:
            cut = idx + len(motif)
            if best_cut is None or cut > best_cut:
                best_cut = cut

    if best_cut is None:
        return None

    vh, vl = s[:best_cut], s[best_cut:]

    if len(vh) < MIN_VH_LEN or len(vl) < MIN_VL_LEN:
        return None
    return vh, vl


def split_vh_vl(line: str):
    s = line.strip().replace(" ", "")
    if not s:
        return None

    # Try VL-start detection first — it is the more reliable of the two strategies.
    res = split_by_vl_start(s)
    if res is not None:
        return res

    # If no VL-start motif is found (e.g. novel or mutated N-termini), fall back
    # to locating the conserved VH C-terminal J-gene sequence.
    res = split_by_heavy_end(s)
    if res is not None:
        return res

    return None


def write_pairs_txt(pairs):
    PAIRS_TXT.write_text(
        "\n".join([f"{vh}, {vl}" for vh, vl in pairs]) + ("\n" if pairs else "")
    )


def write_failed_txt(failed):
    FAILED_TXT.write_text("\n".join(failed) + ("\n" if failed else ""))


def write_pairs_fasta(pairs):
    """
    Writes paired FASTA in the form:
      >ab_1
      VH:VL
    """
    n_written = 0
    with PAIRS_FASTA.open("w") as f:
        for i, (vh, vl) in enumerate(pairs, 1):
            vh = vh.strip().replace(" ", "")
            vl = vl.strip().replace(" ", "")
            if not vh or not vl:
                continue
            # Reject sequences with non-alphabetic characters (e.g. gaps, digits
            # from incomplete splitting or model artefacts).
            if not vh.isalpha() or not vl.isalpha():
                continue
            # ColabFold/AlphaFold-Multimer reads ":" as a chain break; no linker
            # is needed in this format.
            f.write(f">ab_{i}\n{vh}:{vl}\n")
            n_written += 1

    if n_written == 0:
        raise RuntimeError(
            "No valid pairs were written to FASTA. "
            "Check splitting output in pairs_split.txt."
        )
    return n_written


def main():
    if not INFILE.exists():
        raise FileNotFoundError(f"Input file not found: {INFILE}")

    lines = INFILE.read_text().splitlines()

    pairs = []
    failed = []

    for line in lines:
        res = split_vh_vl(line)
        if res is None:
            failed.append(line)
        else:
            pairs.append(res)

    write_pairs_txt(pairs)
    write_failed_txt(failed)
    n_fasta = write_pairs_fasta(pairs)

    print(f"Input file : {INFILE}")
    print(f"Total lines: {len(lines)}")
    print(f"Split OK   : {len(pairs)}")
    print(f"Failed     : {len(failed)}")
    print(f"Pairs txt  : {PAIRS_TXT}")
    print(f"Failed txt : {FAILED_TXT}")
    print(f"Pairs fasta: {PAIRS_FASTA}  (entries written: {n_fasta})")

    if pairs:
        vh, vl = pairs[0]
        print("\nFirst split example:")
        print(f"VH length: {len(vh)} | starts: {vh[:10]} ... ends: {vh[-10:]}")
        print(f"VL length: {len(vl)} | starts: {vl[:10]} ... ends: {vl[-10:]}")


if __name__ == "__main__":
    main()

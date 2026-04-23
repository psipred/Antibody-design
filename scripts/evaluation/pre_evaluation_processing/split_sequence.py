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
    "DIQMTQ", "DIVMTQ", "DVVMTQ",
    "EIVLTQ", "EIVMTQ",
    "QSVLTQ", "QSALTQ", "QITLK", "QVTLR",
    "ENVLTQ", "SYELTQ", "SSELTQ", "AIQLTQ",
    "MTQSP", "VLTQSP", "LTQSP",
]

# Fallback heavy-end motifs (broad)
HEAVY_END_MOTIFS = [
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
    safe_offset = 80
    candidates = []
    for motif in VL_START_MOTIFS:
        idx = s.find(motif, safe_offset)
        if idx != -1:
            candidates.append(idx)

    if not candidates:
        return None

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
        idx = s.rfind(motif)
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

    res = split_by_vl_start(s)
    if res is not None:
        return res

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
            if not vh.isalpha() or not vl.isalpha():
                continue
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
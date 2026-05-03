#!/usr/bin/env python2
"""
Batch CABSflex runner for ColabFold multimer outputs
=====================================================
Discovers rank_001 PDB files produced by ColabFold (best-confidence model per
antibody), then submits them as CABSflex flexibility simulations in parallel.
Results land in OUTPUT_DIR/<job_id>/ with a per-run log and a batch summary TSV.

CABSflex simulation strategy
-----------------------------
* Temperature 1.4–1.4 (Lennard-Jones units): near-native range that preserves
  the overall fold while allowing loop breathing.
* `-g all 3 3.8 8.0`: applies distance restraints to ALL secondary-structure
  elements (helices and sheets) but leaves loops — including CDR H3 — completely
  free to move.  This ensures that the RMSF signal we measure for H3 reflects
  genuine loop flexibility, not rigid-body rotation of the whole framework.
* `--weighted-fit gauss`: superimposes each sampled conformation onto the
  reference by weighting on secondary-structure elements (Gaussian weighting).
  This removes global tumbling from the RMSF, so high values specifically mark
  regions that move *relative to the framework* (i.e. the CDR loops).
* REPLICAS=1: a single replica is sufficient for computing relative RMSF values
  across many antibodies; multiple replicas would be needed for absolute
  thermodynamic observables.

Inputs
------
  INPUT_DIR  : directory of ColabFold output folders (one per antibody),
               each containing at least one *_unrelaxed_rank_001_*.pdb.
               Also handles a flat layout where all PDBs sit directly in
               INPUT_DIR without subdirectories.

Outputs
-------
  OUTPUT_DIR/<job_id>/  : CABSflex work directories (trajectory, RMSF.csv, etc.)
  OUTPUT_DIR/batch_summary.tsv : per-job success/failure and wall-clock time

"""

from __future__ import print_function

import glob
import logging
import os
import re
import subprocess
import sys
import time
from multiprocessing import Pool

# ---------------------------------------------------------------------------
# CONFIGURE THESE
# ---------------------------------------------------------------------------

INPUT_DIR  = "/home/alanwu/Documents/colabfold pipeline/colabfold outputs folder/iggen"
OUTPUT_DIR = "/home/alanwu/Documents/iggen_model/evaluation_metrics/cabsflex/iggen"
WORKERS    = 6   # parallel jobs (6 out of 8 cores, leaves 2 free)

# CABSflex simulation parameters
MC_CYCLES  = 50
MC_STEPS   = 50
REPLICAS   = 1    # single replica for speed; sufficient for relative RMSF comparison
NUM_MODELS = 10   # representative output models per run

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cabsflex_batch")

# ---------------------------------------------------------------------------
# Find rank_001 PDBs
# ---------------------------------------------------------------------------

RANK001_PATTERN = re.compile(r".*_unrelaxed_rank_001_.*\.pdb$")


def find_rank001_pdbs(input_dir):
    """
    Searches input_dir for rank_001 PDB files.
    Handles two layouts:
      1. Flat:   input_dir/ab_1_unrelaxed_rank_001_*.pdb
      2. Nested: input_dir/ab_1/ab_1_unrelaxed_rank_001_*.pdb
    Returns list of (job_id, pdb_path) tuples.
    """
    jobs = []

    # First try nested (one subdir per antibody)
    for entry in sorted(os.listdir(input_dir)):
        subdir = os.path.join(input_dir, entry)
        if os.path.isdir(subdir):
            candidates = [
                f for f in glob.glob(os.path.join(subdir, "*.pdb"))
                if RANK001_PATTERN.match(os.path.basename(f))
            ]
            if candidates:
                jobs.append((entry, candidates[0]))

    # If nothing found nested, try flat layout
    if not jobs:
        candidates = [
            f for f in glob.glob(os.path.join(input_dir, "*.pdb"))
            if RANK001_PATTERN.match(os.path.basename(f))
        ]
        for pdb in sorted(candidates):
            job_id = os.path.basename(pdb).replace(".pdb", "")
            jobs.append((job_id, pdb))

    return jobs


# ---------------------------------------------------------------------------
# Run a single CABSflex job
# ---------------------------------------------------------------------------

def run_job(args):
    job_id, pdb_path, outdir = args

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    log_path = os.path.join(outdir, "cabsflex.log")

    cmd = [
        "CABSflex",
        "-i", pdb_path,
        "--work-dir", outdir,
        "-a", "20",            # MC annealing cycles (default)
        "-y", str(MC_CYCLES),  # MC cycles
        "-s", str(MC_STEPS),   # MC steps between frames
        "-r", str(REPLICAS),   # replicas for better loop sampling
        "-t", "1.4", "1.4",    # near-native temperature (start, end in CABS LJ units)
        # Restrain secondary-structure elements; gap parameters (3, 3.8, 8.0)
        # are SS detection thresholds — loops including CDR H3 remain unrestrained.
        "-g", "all", "3", "3.8", "8.0",
        "-k", str(NUM_MODELS), # number of clustered output models
        # Superimpose on SS elements (Gaussian weighting) so CDR H3 RMSF
        # captures loop movement relative to the antibody framework.
        "--weighted-fit", "gauss",
        "-M",                  # generate contact maps
    ]

    start = time.time()
    try:
        # Redirect both stdout and stderr to a per-job log; keeps the terminal
        # output clean and allows post-hoc debugging of individual failures.
        with open(log_path, "w") as lf:
            subprocess.check_call(cmd, stdout=lf, stderr=subprocess.STDOUT)
        elapsed = time.time() - start
        return (job_id, True, elapsed, "")
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        return (job_id, False, elapsed, "Return code {}. See {}".format(e.returncode, log_path))
    except Exception as e:
        elapsed = time.time() - start
        return (job_id, False, elapsed, str(e))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(results, output_dir):
    summary_path = os.path.join(output_dir, "batch_summary.tsv")
    with open(summary_path, "w") as f:
        f.write("job_id\tstatus\telapsed_s\tmessage\n")
        for job_id, success, elapsed, message in sorted(results, key=lambda x: x[0]):
            status = "OK" if success else "FAILED"
            f.write("{}\t{}\t{:.1f}\t{}\n".format(job_id, status, elapsed, message))
    return summary_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dry_run = "--dry-run" in sys.argv
    workers = WORKERS

    # Parse optional --workers N from command line
    for i, arg in enumerate(sys.argv):
        if arg == "--workers" and i + 1 < len(sys.argv):
            workers = int(sys.argv[i + 1])

    # Find jobs
    log.info("Scanning: {}".format(INPUT_DIR))
    jobs = find_rank001_pdbs(INPUT_DIR)

    if not jobs:
        log.error("No rank_001 PDB files found in {}".format(INPUT_DIR))
        sys.exit(1)

    log.info("Found {} jobs".format(len(jobs)))

    # Build job args
    job_args = []
    for job_id, pdb_path in jobs:
        outdir = os.path.join(OUTPUT_DIR, job_id)
        job_args.append((job_id, pdb_path, outdir))
        if dry_run:
            print("DRY RUN: CABSflex -i \"{}\" --work-dir \"{}\"".format(pdb_path, outdir))

    if dry_run:
        log.info("Dry run complete. {} jobs would be submitted.".format(len(job_args)))
        return

    # Create output root
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    log.info("Running {} jobs with {} workers...".format(len(job_args), workers))

    # imap_unordered streams results as they finish rather than buffering all
    # completions; this allows progress logging even when jobs have unequal runtimes.
    pool = Pool(processes=workers)
    results = []
    completed = 0

    for result in pool.imap_unordered(run_job, job_args):
        job_id, success, elapsed, message = result
        results.append(result)
        completed += 1
        status = "OK" if success else "FAILED"
        log.info("[{}/{}] {} {} ({:.1f}s){}".format(
            completed, len(job_args), status, job_id, elapsed,
            " -- " + message if message else ""
        ))

    pool.close()
    pool.join()

    # Write summary
    n_ok   = sum(1 for r in results if r[1])
    n_fail = len(results) - n_ok
    summary_path = write_summary(results, OUTPUT_DIR)

    log.info("=" * 60)
    log.info("Done. {} succeeded, {} failed.".format(n_ok, n_fail))
    log.info("Summary: {}".format(summary_path))

    if n_fail:
        log.warning("Failed jobs:")
        for job_id, success, elapsed, message in results:
            if not success:
                log.warning("  {}: {}".format(job_id, message))


if __name__ == "__main__":
    main()

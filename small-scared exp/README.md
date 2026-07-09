```markdown
# Multi-Step Verification for Dual Lattice Attacks

This repository contains a modified version of the experimental verification code originally developed by [Kevin Carrier](https://github.com/kevin-carrier/CodedDualAttack) in the [ScoreExperimentalDistribution](https://github.com/kevin-carrier/CodedDualAttack/tree/main/verifyModel/ScoreExperimentalDistribution) folder.

We have refactored the original pipeline into **multiple independent steps** to allow fine-grained control and easier debugging. The code also relies on external lattice reduction and sieving tools.

## Dependencies

Before running the scripts, please install the following tools:

- **[BLASter](https://github.com/ludopulles/BLASter)** – used for BKZ reduction.
- **[g6k](https://github.com/fplll/g6k/)** – used for lattice sieving (progressive sieve).

Make sure both are compiled and available on your system. All paths in the scripts (e.g., to input/output files) are **user-configurable** – you will need to adjust them according to your local directory structure.

## Step-by-Step Instructions

### 1. Generate Dual Lattice Bases

Run `build_dual.py` to create lattice bases of different dimensions.

```bash
python build_dual.py
```

You can modify the parameters (dimensions, number of samples, etc.) directly inside the script.

### 2. BKZ Reduction with BLASter

After generating the basis (typically saved as `basis.txt`), reduce it using BLASter's progressive BKZ algorithm. For example, to perform a BKZ-60 reduction with 32 threads:

```bash
./BLASter/src/app.py -v -j 32 -i ./basis.txt -o ./bkz_output_dual.txt -t 1 -b60 -P2
```

Adjust the paths (`-i`, `-o`) and the beta value (`-b`) as needed.

### 3. Sieving with g6k

Once the reduced basis is obtained, run the progressive sieve provided by g6k. The script `sieve.py` (put it inside the g6k directory) reads the BKZ‑output basis and produces the final sieve statistics.

```bash
cd /path/to/g6k
python sieve.py
```

> **Note**: Some parts of the sieving code were taken from Ludo Pulles’ work in  
> [AccurateScorePredictionDualSieveAttacks](https://github.com/ludopulles/AccurateScorePredictionDualSieveAttacks/blob/main/code/utils.py).

### 4. Run the Small-Scale Experiments

Two experimental scripts are provided to test the **MS‑LSC** and **LSC‑MS** algorithms on small instances:

- `Algorithm1_ms.py`
- `Algorithm1_lsc_ms.py`

Run them sequentially:

```bash
python Algorithm1_ms.py
python Algorithm1_lsc_ms.py
```

You can modify internal parameters (e.g., dimension of sample, number of short vectors) directly in each file.

## Important Notes

- **File Paths:** All input/output file paths are hard-coded in the scripts. You **must** change them to match your actual folder structure before running anything.
- **Dependencies:** Both BLASter and g6k must be installed and their executables/scripts accessible from the command line. The example commands assume you are in the root directory of each tool.

## Acknowledgments

- **Original optimization pipeline:** This code is adapted from the work by [Kevin Carrier](https://github.com/kevin-carrier/CodedDualAttack) – see [CodedDualAttack/verifyModel/ScoreExperimentalDistribution](https://github.com/kevin-carrier/CodedDualAttack/tree/main/verifyModel/ScoreExperimentalDistribution).
- **Sieving utilities:** Some parts of the sieving code were inspired by/borrowed from [Ludo Pulles](https://github.com/ludopulles) in his repository [AccurateScorePredictionDualSieveAttacks](https://github.com/ludopulles/AccurateScorePredictionDualSieveAttacks/blob/main/code/utils.py).
- **External tools:** We thank the developers of [BLASter](https://github.com/ludopulles/BLASter) and [g6k](https://github.com/fplll/g6k/) for making their state‑of‑the‑art lattice reduction and sieving implementations publicly available.

---

For any questions or issues, please adjust the scripts according to your environment and refer to the original repositories for detailed documentation.
```

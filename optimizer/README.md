```markdown
# Optimizer for Coded Dual Attack Complexity

This repository contains a minimal adaptation of the optimal complexity search program originally developed by [Kevin Carrier](https://github.com/kevin-carrier/CodedDualAttack). The original implementation can be found at:

[https://github.com/kevin-carrier/CodedDualAttack/tree/main/OptimizeCodedDualAttack](https://github.com/kevin-carrier/CodedDualAttack/tree/main/OptimizeCodedDualAttack)

This project includes only the files necessary to run the optimizer. All credit for the core algorithm and implementation belongs to the original author.

## Requirements

- [SageMath](https://www.sagemath.org/) environment (which provides `sage-python`)

## Quick Start

### 1. Extract the estimator

Unzip the `estimator.zip` archive in the same directory as the scripts.

```bash
unzip estimator.zip
```

### 2. Run the optimizer

Execute the optimizer with:

```bash
sage-python optimizer_naive.py
```

### 3. Configure the modulus `p`

To change the modulus `p`, edit **line 128** of `optimizer_naive.py` and set the desired value.

Supported values for `p` are:

- 512
- 1024
- 2048
- 3072
- 3329

### 4. View the results

The optimization results are saved to:

```
optimized_withoutExperimentalPolar.pkl
```

To read and display the results, run:

```bash
sage-python read.py
```

> **Note:** If your file paths differ from the default structure, please update the paths in the scripts accordingly to match your local environment.


## License
This project is a derivative work based on the [CodedDualAttack](https://github.com/kevin-carrier/CodedDualAttack) repository by Kevin Carrier. Please refer to the original repository for licensing terms.
## Acknowledgments
-Original author: Kevin Carrier
-Original repository: CodedDualAttack

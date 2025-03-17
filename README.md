# SEROAISE: Sequential Estimation of RoA Based on Invariant Set Estimation

This repository implements a novel framework for constructing the Region of Attraction (RoA) for dynamical systems. It supports systems derived from Piecewise Affine (PWA) functions or Neural Networks (NNs) with ReLU activations.

---

## Overview

**SEROAISE** (Sequential Estimation of RoA based on Invariant Set Estimation) computes a Lyapunov-like PWA function over a certified PWA invariant set. Unlike traditional methods that enforce Lyapunov conditions on pre-selected domains, SEROAISE applies these conditions over an invariant subset obtained via the **Iterative Invariant Set Estimator (IISE)**.

A key aspect is the **Non-Uniform Growth of Invariant Set (NUGIS)**, which allows the generation of systematically larger certified invariant sets compared to state-of-the-art approaches. Several examples in this repository illustrate the method’s effectiveness, including applications to dynamical systems derived from learning algorithms.

---
## **Installation and Requirements**

### **1. Clone the Repository**
```bash
git clone https://github.com/PouyaSamanipour/SEROAISE.git
cd SEROIASE
```

### **2. Set Up a Python Virtual Environment**
Create a virtual environment to avoid conflicts with system-wide packages:
```bash
python -m venv seroaise_env
seroaise_env\Scripts\activate
```
## Requirements & Dependencies

This project depends on the following Python packages:

- `numpy`
- `numba`
- `torch`
- `scipy`
- `matplotlib`
- `pandas`
- `pycddlib`

### Additional Dependency: Gurobi

**Gurobi** is required for the optimization components. Please install and configure Gurobi manually by following the instructions on the [Gurobi website](https://www.gurobi.com/). Gurobi is not installed via pip.
---

## **How to Run the Code**

### **1. Navigate to the Project Directory**
```bash
cd SEROIASE
```

### **2. Run Example Scripts**
Example scripts are available in the `Examples` folder. Two examples are provided:
- **Inverted Pendulum**
To run a specific example, use:
```bash
python Examples/IP_large_domain.py
```


### **3. Customize the Framework**
Modify the input parameters in the scripts to test different ReLU-based dynamical systems or PWA representations.


## **License**
This project is free for academic use under the MIT license. Please refer to the `LICENSE` file for more details.


---

## **Contact**
For questions or inquiries, please contact **Pouya Samanipour** at [psa254@uky.edu](mailto:psa254@uky.edu).



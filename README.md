
---

# Adaptive Physics-Informed System Modeling (APSM)

This repository provides the official Python implementation of the paper:

**“Adaptive Physics-Informed System Modeling for Digital Twin of Engineering Structures under Non-Stationary Conditions”**
*Biqi Chen, Ying Wang, Jian Yang, Jinping Ou*

School of Intelligent Civil and Marine Engineering, Harbin Institute of Technology (Shenzhen)

---

## 🔎 Overview

The **Adaptive Physics-Informed System Modeling (APSM)** framework integrates **residual-driven optimization** with **physics constraints** to enable **online system identification** 
and **digital twin modeling** of large-scale civil infrastructure under **non-stationary conditions**. 

Key features of APSM:

* Online updating of system matrices using **Kalman Filter (KF)** and **proximal gradient descent**.
* Robust **full-state reconstruction** from sparse sensor measurements.
* Embedded **physical constraints** for stability, interpretability, and noise robustness.
* High computational efficiency: processing 1 hour of 12-channel 50 Hz bridge data in \~33 seconds.

---

## 📌 Key Contributions

* ✅ Unified framework for **digital twin simulation** and **online model updating**.  
* ✅ Effectively suppresses **daily prediction error oscillations** in long-term bridge monitoring.  
* ✅ Demonstrates superior **accuracy and generalization** compared with traditional offline system identification methods.  
* ✅ Provides a scalable and efficient tool for **real-time structural health monitoring (SHM)** and **control**.  

---

## 📂 Case Studies

The APSM framework has been validated through:

1. **Five-Degree-of-Freedom System** – verifies robustness under noise and sparse measurements.  
2. **Van der Pol Oscillator** – demonstrates capability for nonlinear and time-varying dynamics.  
3. **Hangzhou Bay Bridge** – real-world validation with long-term acceleration monitoring data.  

---

## 📦 Contents

- **Five Degrees of Freedom System1**  
  Numerical simulation of a five-DOF mass–spring–damper system, used in the paper to validate APSM performance under noise and sparse measurements.  

- **Five Degrees of Freedom System2**  
  Comparative study of APSM with and without **physical constraints**, highlighting differences in prediction accuracy and system matrix identification.  

  **Notes**:  
  1. Run the constrained and unconstrained codes separately.  
  2. The random seed is fixed for reproducibility.  
  3. Do not run both scripts simultaneously to avoid inconsistencies in random numbers.  

- **Van der Pol Oscillator**  
  Implementation of a nonlinear oscillator with temperature-dependent stiffness and nonlinear damping. Includes RK4 integration, phase portrait visualization, and system identification using APSM.  

- **APSM_Algorithm**  
  Core implementation of the APSM framework. Includes:  
  * ERA (Eigensystem Realization Algorithm) for initial state-space model construction.  
  * Online updating of system matrices using **Kalman Filter** and **proximal gradient descent**.
  * The default data saving path should be replaced with the user’s own path, and some parameters related to data dimensions may need to be adjusted accordingly.


- **state_data.npy**  
Example state data file for reproducing the Van der Pol Oscillator experiment.

---

## 📝 Notes

- ERA is provided as a default state-space modeling method. Other methods (e.g., **DMD**, **SSI**, **NExT+ERA**) can also be integrated as initial models.  
- Scripts include functions for gradient-based system identification, Frobenius norm error analysis, and visualization of simulation results.  

---



要不要我帮你在 README 里再加一个 **可视化算法流程图（APSM 框架示意图）** 部分？这样仓库首页看起来会更直观、吸引人。


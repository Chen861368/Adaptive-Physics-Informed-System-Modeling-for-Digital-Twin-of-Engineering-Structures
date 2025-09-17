
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

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Chen861368/Adaptive-Physics-Informed-System-Modeling.git
cd Adaptive-Physics-Informed-System-Modeling
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run APSM on sample data

```bash
python run_apsm.py --config configs/example.yaml
```

---



要不要我帮你在 README 里再加一个 **可视化算法流程图（APSM 框架示意图）** 部分？这样仓库首页看起来会更直观、吸引人。


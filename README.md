# AI-Driven Adaptive Communication Equalizer & Automated Verification Platform

[![CI Automation Test Pipeline](https://github.com)](https://github.com)
![Python Version](https://shields.io)
![Test Framework](https://shields.io)

An advanced baseline and industrial-grade software/hardware verification platform for baseband communication. This project benchmarks traditional adaptive filtering algorithms against modern deep learning optimization techniques by integrating a custom, hand-written **Adam optimizer** into a complex-valued channel equalization system.

The entire architecture is partitioned into a clean **DUT (Device Under Test)** and **Test Bench** paradigm, fully integrated with automated regression testing and a GitHub Actions CI/CD pipeline.

---

## 🚀 Key Highlights & Engineering Value
* **Cross-Domain Innovation:** Successfully bridges **Digital Signal Processing (DSP)** and **Machine Learning (ML)** by adapting the Adam optimizer to process complex-valued baseband signals (I/Q channels) instead of standard real-valued time-series.
* **Pure NumPy Implementation:** Built entirely from scratch without high-level ML frameworks (like PyTorch or Scikit-learn). Implemented explicit first/second-order moment matrix updates and full bias correction mechanisms in complex vector spaces (\(\mathbb{C}\)).
* **Modern QA Automation:** Developed a production-standard automated regression testing framework utilizing `pytest` and automated threshold assertions (Spec Limit Checks) for steady-state Performance (MSE), completely isolated to prevent cross-test state pollution.
* **Hardware-Aware Design Insight:** Demonstrates a profound understanding of hardware implementation trade-offs (Computational Complexity vs. Convergence Rate) for IC deployment.

---

## 🛠️ Algorithm Benchmarking & Trade-offs

This platform evaluates three distinct optimization paradigms under identical fading channel environments (SNR = 15dB):

| Algorithm | Optimization Type | Computational Complexity | Physical / Hardware Trade-off |
| :--- | :--- | :--- | :--- |
| **LMS** | Traditional Fixed Step-size | \(\mathcal{O}(M)\) | Extremely low hardware cost, but suffers from slow convergence and poor tracking in highly dynamic channels. |
| **RLS** | Higher-order Recursive | \(\mathcal{O}(M^2)\) | Fastest convergence by tracking inverse auto-correlation matrix, but prohibitive chip area/power consumption due to matrix multiplications. |
| **Adam (AI)** | Adaptive Moment Estimation | \(\mathcal{O}(M)\) | **Optimal Balance.** Reaches RLS-like fast convergence speeds while maintaining an LMS-level linear computational profile. Highly suitable for hardware deployment. |

---

## 🔬 System Architecture

Following industrial automated software verification standards, the repository is structured into a separated Device Under Test (DUT) and Test Bench architecture:

```text
AI-Driven-Adaptive-Communication-Equalizer/
├── .github/workflows/ci.yml   # GitHub Actions Automation CI Pipeline
├── src/
│   ├── __init__.py            # Python Package Marker
│   └── equalizer_dut.py       # Device Under Test (DUT) Core Core Module
└── tests/
    └── test_equalizer_qa.py   # Test Bench (Pytest Regression Suite)
```

### 1. Device Under Test (DUT) (`src/equalizer_dut.py`)
Processes incoming complex baseband symbols using a sliding window convolution form. For the **Adam Optimizer mode**, the complex-valued update rules are derived from first principles as follows:

* **Gradient Direction**:
$$g_t = -e_t^* \cdot \mathbf{x}_t$$

* **First-order Momentum:** $$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$ (Tracks gradient velocity)

* **Second-order Momentum:** $$v_t = \beta_2 v_{t-1} + (1-\beta_2)|g_t|^2$$ (Tracks gradient power, enforced in real-domain via absolute squares)

* **Bias Correction**: (Prevents bias from tending towards zero)
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}$$

* **Weights Update Matrix:**
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\alpha \cdot \hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

### 2. QA Test Bench (`tests/test_equalizer_qa.py`)
* **Signal Synthesis:** Generates standard QPSK constellations normalized by$\frac{1}{\sqrt{2}}$.
* **Channel Modeling:** Convolves input symbols with a multi-path frequency-selective fading channel vector: `[1.0, 0.4 - 0.3j, 0.1 + 0.2j]` and injects multi-variable AWGN noise.
* **Deterministic Verification & Test Isolation:** Enforces explicit random seeds (`np.random.seed(42)`) and encapsulates data via Pytest Fixtures. This eliminates cross-test state contamination, allowing each algorithm to be evaluated in an identical, pristine memory state.
* **Automated Assertion:** Evaluates the mean of the final 500 steady-state Mean Squared Error (MSE) data points against a rigid `spec_limit` threshold (0.1) to trigger automated `PASS`/`FAIL` verification flags.

---

## 💻 Installation & Quick Start

Ensure you have a Python environment setup.

### Installation
```bash
pip install -r requirements.txt
```

### Run Industrial Automated QA Tests (Pytest)
To execute the encapsulated verification regression with verbose printouts enabled, run:
```bash
python -m pytest tests/ -v -s
```

### Expected Console Output
```text
tests/test_equalizer_qa.py::test_equalizer_regression[LMS] PASSED
[Pytest CI] LMS -> Steady-state MSE: 0.09014

tests/test_equalizer_qa.py::test_equalizer_regression[RLS] PASSED
[Pytest CI] RLS -> Steady-state MSE: 0.08843

tests/test_equalizer_qa.py::test_equalizer_regression[Adam] PASSED
[Pytest CI] Adam -> Steady-state MSE: 0.09157
```

---

## 🚀 Continuous Integration (CI/CD)
This repository is configured with a automated GitHub Actions workflow (`.github/workflows/ci.yml`). Upon every `push` or `pull_request` to the `main` branch, an ephemeral Linux container is automatically provisioned to setup environment configurations, cache `pip` dependencies, set the explicit `PYTHONPATH` boundaries, and trigger the `pytest` matrix. This ensures full code verification and guarantees that no algorithm modifications break the algorithmic steady-state specifications.

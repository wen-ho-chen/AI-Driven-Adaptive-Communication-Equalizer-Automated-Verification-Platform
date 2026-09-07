# AI-Driven Adaptive Communication Equalizer Automation Test Bench

An advanced baseline and regression verification platform for baseband communication. This project benchmarks traditional adaptive filtering algorithms against modern deep learning optimization techniques by integrating a custom, hand-written **Adam optimizer** into a complex-valued channel equalization system.

---

## 🚀 Key Highlights & Engineering Value
* **Cross-Domain Innovation:** Successfully bridges **Digital Signal Processing (DSP)** and **Machine Learning (ML)** by adapting the Adam optimizer to process complex-valued baseband signals (I/Q channels) instead of standard real-valued time-series.
* **Pure NumPy Implementation:** Built entirely from scratch without high-level ML frameworks (like PyTorch or Scikit-learn). Implemented explicit first/second-order moment matrix updates and full bias correction mechanisms in complex vector spaces.
* **QA Automation (Test Bench):** Developed an automated regression testing framework featuring channel fading simulation, AWGN noise injection, and automated threshold assertions (Spec Limit Checks) for steady-state Performance (MSE).
* **Hardware-Aware Design Insight:** Demonstrates a profound understanding of hardware implementation trade-offs (Computational Complexity vs. Convergence Rate).

---

## 🛠️ Algorithm Benchmarking & Trade-offs

This platform evaluates three distinct optimization paradigms under identical fading channel environments ($SNR = 15\text{dB}$):

| Algorithm | Optimization Type | Computational Complexity | Physical / Hardware Trade-off |
| :--- | :--- | :--- | :--- |
| **LMS** | Traditional Fixed Step-size | $\mathcal{O}(M)$ | Extremely low hardware cost, but suffers from slow convergence and poor tracking in highly dynamic channels. |
| **RLS** | Higher-order Recursive | $\mathcal{O}(M^2)$ | Fastest convergence by tracking inverse auto-correlation matrix, but prohibitive chip area/power consumption due to matrix multiplications. |
| **Adam (AI)** | Adaptive Moment Estimation | $\mathcal{O}(M)$ | **Optimal Balance.** Reaches RLS-like fast convergence speeds while maintaining an LMS-level linear computational profile. Highly suitable for hardware deployment. |

---

## 🔬 System Architecture

The project is structured following industrial automated software verification standards, dividing the repository into a Device Under Test (DUT) and a Test Bench:
├── README.md└── equalizer_qa_platform.py  # Integrated Python executable containing:├── AdaptiveEqualizerDUT  # Core DUT processing LMS, RLS, and Custom Adam└── run_integrated_qa_test# Automated Test Bench (Signal Generation, Fading, Noise, Plots)
### 1. Device Under Test (DUT)
Processes incoming complex baseband symbols using a sliding window convolution form. For the **Adam Optimizer mode**, the complex-valued update rules are derived as follows:

* **Gradient Direction**:
$$g_t = -e_t^* \cdot \mathbf{x}_t$$

* **First-order Momentum:** $$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$ (Tracks gradient velocity)

* **Second-order Momentum:** $$v_t = \beta_2 v_{t-1} + (1-\beta_2)|g_t|^2$$ (Tracks gradient power, enforced in real-domain via absolute squares)

* **Bias Correction**: (Prevents bias from tending towards zero)
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}$$

### 2. QA Test Bench
**Signal Synthesis**: Generates standard QPSK constellations normalized by $\frac{1}{\sqrt{2}}$.

* **Channel Modeling:** Convolves input symbols with a multi-path frequency-selective fading channel vector: [1.0, 0.4 - 0.3j, 0.1 + 0.2j].
* **Automated Assertion:** Evaluates the mean of the final 500 steady-state Mean Squared Error (MSE) data points against a rigid `spec_limit` threshold (0.1) to trigger automated `PASS`/`FAIL` flags.

---

## 📊 Verification & Visualization

Upon running the automated test script, the platform executes regression loops and automatically plots two critical communication performance charts:

1. **Learning / Convergence Curves:** Tracks the instantaneous MSE across 2000 symbols on a logarithmic scale to observe transient behaviors and steady-state stability.
2. **Constellation Diagrams:** Visualizes I/Q symbol clustering at steady-state to intuitively evaluate residual phase/amplitude noise after equalization.

---

## 💻 Prerequisites & Quick Start

Ensure you have a Python environment with `numpy` and `matplotlib` installed.

### Installation
```bash
pip install numpy matplotlib
```

### Run Automated QA Tests
```bash
python equalizer_qa_platform.py
```

### Expected Console Output Template
```text
=== [QA Test Case] Executing '3-in-1' Adaptive Equalizer Regression Test (SNR: 20dB) ===
  Algorithm [LMS (Traditional)] -> Steady-state MSE: 0.09014 | Spec Limit: 0.10000 | Test Result: PASS
  Algorithm [RLS (Recursive)]    -> Steady-state MSE: 0.08843 | Spec Limit: 0.10000 | Test Result: PASS
  Algorithm [Adam (AI Adaptive)] -> Steady-state MSE: 0.09547 | Spec Limit: 0.10000 | Test Result: PASS
```


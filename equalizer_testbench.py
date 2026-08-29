import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# 1. 被測系統 (DUT: Device Under Test)
# ==========================================
def adaptive_equalizer_dut(rx_symbols, tx_pilots, algo="LMS", filter_order=5):
    """模擬待測的自適應濾波器演算法。

    QA 測試重點：驗證不同演算法在相同輸入下的穩態效能與收斂軌跡。
    """
    num_symbols = len(rx_symbols)
    w = np.zeros(filter_order, dtype=complex)
    y_out = np.zeros(num_symbols, dtype=complex)
    mse = np.zeros(num_symbols)

    if algo == "RLS":
        delta = 0.1
        P = np.eye(filter_order, dtype=complex) / delta
        rls_lambda = 0.99

    lms_mu = 0.05

    for t in range(filter_order, num_symbols):
        x = rx_symbols[t - filter_order + 1 : t + 1][::-1]

        if algo == "LMS":
            y = np.dot(w.conj(), x)
            e = tx_pilots[t] - y
            w += lms_mu * e.conj() * x
        elif algo == "RLS":
            y = np.dot(w.conj(), x)
            e = tx_pilots[t] - y
            Px = np.dot(P, x)
            k = Px / (rls_lambda + np.dot(x.conj(), Px))
            w += k * e.conj()
            P = (
                P - np.dot(k[:, None], np.dot(x.conj(), P)[None, :])
            ) / rls_lambda

        y_out[t] = y
        mse[t] = np.abs(e) ** 2

    return y_out, mse


# ==========================================
# 2. QA 自動化測試驗證平台與數據蒐集 (Test Bench)
# ==========================================
def run_qa_regression_test_with_plots():
    """QA 迴歸測試核心：模擬環境、自動驗證，並整合雙場景與星座圖的視覺化。"""
    np.random.seed(42)
    num_symbols = 2000
    standard_channel = np.array([1.0, 0.4 - 0.3j, 0.1 + 0.2j])

    # 產生 QPSK 測試訊號
    bits = np.random.randint(0, 2, num_symbols * 2)
    qpsk_mapping = {
        (0, 0): 1 + 1j,
        (0, 1): -1 + 1j,
        (1, 1): -1 - 1j,
        (1, 0): 1 - 1j,
    }
    tx_symbols = (
        np.array(
            [
                qpsk_mapping[(bits[i], bits[i + 1])]
                for i in range(0, len(bits), 2)
            ]
        )
        / np.sqrt(2)
    )

    # 定義要測試的兩種 QA 場景 [SNR_dB, Spec_Limit, 場景名稱]
    scenarios = [
        (25, 0.05, "Scenario 1: Standard Environment (25dB)"),
        (5, 0.5, "Scenario 2: Stress Test (5dB)"),
    ]

    # 初始化繪圖畫布 (2列2行)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (snr_db, spec_limit, title) in enumerate(scenarios):
        # 模擬通道干擾與加性高斯白雜訊 (AWGN)
        rx_faded = np.convolve(tx_symbols, standard_channel, mode="same")
        sig_power = np.mean(np.abs(rx_faded) ** 2)
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols)
        )
        rx_symbols = rx_faded + noise

        print(f"=== [QA Test Case] 執行 {title} ===")

        # 執行測試
        y_out_lms, mse_lms = adaptive_equalizer_dut(
            rx_symbols, tx_symbols, algo="LMS"
        )
        y_out_rls, mse_rls = adaptive_equalizer_dut(
            rx_symbols, tx_symbols, algo="RLS"
        )

        # QA 斷言驗證 (取最後 500 點穩態平均)
        for algo, mse_data in [("LMS", mse_lms), ("RLS", mse_rls)]:
            steady_state_mse = np.mean(mse_data[-500:])
            status = "PASS" if steady_state_mse < spec_limit else "FAIL"
            print(
                f"  算法 [{algo}] -> 穩態 MSE: {steady_state_mse:.5f} | 規格上限: {spec_limit} | 測試結果: {status}"
            )

        # ----------------------------------------
        # 子圖 A：學習收斂曲線 (Learning Curve)
        # ----------------------------------------
        ax_curve = axs[idx, 0]
        ax_curve.plot(mse_lms, label="LMS (mu=0.05)", color="blue", alpha=0.7)
        ax_curve.plot(
            mse_rls, label="RLS (lambda=0.99)", color="red", alpha=0.7
        )
        ax_curve.axhline(
            y=spec_limit,
            color="green",
            linestyle="--",
            label=f"Spec Limit ({spec_limit})",
        )
        ax_curve.set_yscale("log")
        ax_curve.set_title(f"{title} - 收斂曲線")
        ax_curve.set_xlabel("Symbols")
        ax_curve.set_ylabel("MSE (Log Scale)")
        ax_curve.grid(True, which="both", linestyle=":", alpha=0.5)
        ax_curve.legend()

        # ----------------------------------------
        # 子圖 B：等化後星座圖 (Constellation Diagram)
        # ----------------------------------------
        ax_const = axs[idx, 1]
        # 只繪製最後 500 點已收斂的訊號，以便看清穩態散點群聚效果
        ax_const.scatter(
            rx_symbols[-500:].real,
            rx_symbols[-500:].imag,
            color="gray",
            alpha=0.4,
            s=10,
            label="Before (Rx)",
        )
        ax_const.scatter(
            y_out_lms[-500:].real,
            y_out_lms[-500:].imag,
            color="blue",
            alpha=0.6,
            s=8,
            label="After LMS",
        )
        ax_const.scatter(
            y_out_rls[-500:].real,
            y_out_rls[-500:].imag,
            color="red",
            alpha=0.6,
            s=8,
            label="After RLS",
        )

        ax_const.set_title(f"{title} - 穩態星座圖對比")
        ax_const.set_xlabel("In-Phase (I)")
        ax_const.set_ylabel("Quadrature (Q)")
        ax_const.axis("equal")
        ax_const.grid(True, linestyle=":", alpha=0.5)
        ax_const.legend()

        print("\n")

    plt.tight_layout()
    plt.show()


# ==========================================
# 3. 執行完整 QA 整合測試
# ==========================================
if __name__ == "__main__":
    run_qa_regression_test_with_plots()


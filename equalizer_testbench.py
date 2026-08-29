import numpy as np

# 1. 被測系統 (DUT: Device Under Test)

def adaptive_equalizer_dut(rx_symbols, tx_pilots, algo="LMS", filter_order=5):
    """模擬待測的自適應濾波器演算法。
    驗證不同演算法在相同輸入下的穩態效能。
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
            P = (P - np.dot(k[:, None], np.dot(x.conj(), P)[None, :])) / rls_lambda

        y_out[t] = y
        mse[t] = np.abs(e) ** 2

    return y_out, mse


# ==========================================
# 2. QA 自動化測試驗證平台 (Test Bench)
# ==========================================
def run_qa_regression_test(snr_db, channel_h):
    """迴歸測試：模擬環境並自動驗證演算法指標是否 Pass."""
    np.random.seed(42)
    num_symbols = 2000

    # 產生測試訊號 (QPSK)
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

    # 注入通道干擾與雜訊 (模擬環境)
    rx_faded = np.convolve(tx_symbols, channel_h, mode="same")
    sig_power = np.mean(np.abs(rx_faded) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols)
    )
    rx_symbols = rx_faded + noise

    print(
        f"=== [QA Test Case] 執行環境測試 - SNR: {snr_db}dB, 通道階數: {len(channel_h)} ==="
    )

    # 執行測試並驗證 (Assertion)
    for algo in ["LMS", "RLS"]:
        y_out, mse = adaptive_equalizer_dut(rx_symbols, tx_symbols, algo=algo)

        # 評估指標：取最後 500 點穩態的平均 MSE
        steady_state_mse = np.mean(mse[-500:])

        # QA 定義的測試合格門檻 (Threshold)
        # 正常情況下，25dB SNR 時 MSE 應低於 0.05
        spec_limit = 0.05 if snr_db >= 20 else 0.5

        # 斷言機制
        status = "PASS" if steady_state_mse < spec_limit else "FAIL"

        print(
            f"  算法 [{algo}] -> 穩態 MSE: {steady_state_mse:.5f} | 規格上限: {spec_limit} | 測試結果: {status}"
        )


# ==========================================
# 3. 執行 QA 測試場景 (Test Scenarios)
# ==========================================
if __name__ == "__main__":
    # 場景一：正常標準環境測試 (Sanity Test)
    standard_channel = np.array([1.0, 0.4 - 0.3j, 0.1 + 0.2j])
    run_qa_regression_test(snr_db=25, channel_h=standard_channel)

    print("\n")

    # 場景二：壓力測試/惡劣環境 (Corner Case / Stress Test)
    # 測試在低信噪比 (5dB) 下，演算法是否會嚴重發散或異常
    run_qa_regression_test(snr_db=5, channel_h=standard_channel)

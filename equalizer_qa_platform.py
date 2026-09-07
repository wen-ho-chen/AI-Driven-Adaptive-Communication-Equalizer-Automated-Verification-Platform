import math
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 被測系統 (DUT): 傳統、遞迴與 AI 最佳化器的自適應等化器
# ==========================================
class AdaptiveEqualizerDUT:
    """
    自適應通訊等化器核心模組。
    整合：傳統基頻 (LMS)、高階遞迴 (RLS) 與現代深度學習最佳化器 (Adam)。
    """
    def __init__(self, filter_order=11, lr=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-8):
        self.filter_order = filter_order
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        
    def equalize(self, rx_symbols, tx_pilots, algo="LMS"):
        num_symbols = len(rx_symbols)
        # 初始化濾波器權重 (複數矩陣)
        w = np.zeros(self.filter_order, dtype=complex)
        y_out = np.zeros(num_symbols, dtype=complex)
        mse = np.zeros(num_symbols)

        # 【RLS 狀態變數初始化】
        if algo == "RLS":
            delta = 0.1
            P = np.eye(self.filter_order, dtype=complex) / delta
            rls_lambda = 0.99

        # 【Adam 狀態變數初始化】
        if algo == "Adam":
            m_t = np.zeros(self.filter_order, dtype=complex)
            v_t = np.zeros(self.filter_order, dtype=float)  # 功率方差為實數
            t = 0

        # 固定步長的 LMS 參數
        lms_mu = 0.01

        # 滑動視窗 (Sliding Window) 處理基頻訊號
        for idx in range(self.filter_order, num_symbols):
            # 取出當前視窗內的接收訊號 (倒序以符合捲積形式)
            x = rx_symbols[idx - self.filter_order + 1 : idx + 1][::-1]
            
            # 前向傳播：估計等化後的輸出符元
            y = np.dot(w.conj(), x)
            y_out[idx] = y
            
            # 計算複數誤差 (期望導頻 - 等化輸出)
            e = tx_pilots[idx] - y
            mse[idx] = np.abs(e) ** 2
            
            # 依據演算法進行權重更新
            if algo == "LMS":
                w += lms_mu * e.conj() * x
                
            elif algo == "RLS":
                Px = np.dot(P, x)
                k = Px / (rls_lambda + np.dot(x.conj(), Px))
                w += k * e.conj()
                P = (P - np.dot(k[:, None], np.dot(x.conj(), P)[None, :])) / rls_lambda
                
            elif algo == "Adam":
                t += 1
                g_t = -e.conj() * x  # 複數梯度方向
                
                # 一階動量更新 (追蹤梯度方向)
                m_t = self.beta_1 * m_t + (1 - self.beta_1) * g_t
                # 二階動量更新 (追蹤梯度功率)
                v_t = self.beta_2 * v_t + (1 - self.beta_2) * (np.abs(g_t) ** 2)
                
                # 偏差修正 (Bias Correction)
                m_cap = m_t / (1 - (self.beta_1 ** t))
                v_cap = v_t / (1 - (self.beta_2 ** t))
                
                # 權重更新：結合 Adam 動態步長
                w -= (self.lr * m_cap) / (np.sqrt(v_cap) + self.epsilon)

        return y_out, mse

# ==========================================
# 2. QA 自動化測試驗證平台與數據蒐集 (Test Bench)
# ==========================================
def run_integrated_qa_test():
    """QA 迴歸測試核心：驗證 LMS、RLS 與 Adam 在通訊解調上的穩態表現、收斂軌跡與計算性價比。"""
    np.random.seed(42)
    num_symbols = 2000
    
    # 模擬多徑衰落通道 (Standard Channel)
    fading_channel = np.array([1.0, 0.4 - 0.3j, 0.1 + 0.2j])

    # 1. 產生 QPSK 測試訊號
    bits = np.random.randint(0, 2, num_symbols * 2)
    qpsk_mapping = {(0, 0): 1+1j, (0, 1): -1+1j, (1, 1): -1-1j, (1, 0): 1-1j}
    tx_symbols = np.array([qpsk_mapping[(bits[i], bits[i+1])] for i in range(0, len(bits), 2)]) / np.sqrt(2)

    # 2. 建立通訊干擾與雜訊環境 (20dB 中高信噪比環境)
    snr_db = 15
    rx_faded = np.convolve(tx_symbols, fading_channel, mode="same")
    sig_power = np.mean(np.abs(rx_faded) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols))
    rx_symbols = rx_faded + noise

    # 3. 測試規格斷言門檻上限 (Spec Limit)
    spec_limit = 0.1
    
    # 4. 初始化等化器 DUT
    equalizer = AdaptiveEqualizerDUT(filter_order=5, lr=0.01)
    
    print(f"=== [QA Test Case] 執行『三合一』自適應等化器迴歸測試 (SNR: {snr_db}dB) ===")

    # 執行三種演算法
    y_out_lms, mse_lms = equalizer.equalize(rx_symbols, tx_symbols, algo="LMS")
    y_out_rls, mse_rls = equalizer.equalize(rx_symbols, tx_symbols, algo="RLS")
    y_out_adam, mse_adam = equalizer.equalize(rx_symbols, tx_symbols, algo="Adam")

    # 5. QA 自動化自動斷言驗證 (Assertion Check)
    algos_to_test = [("LMS (傳統固定步長)", mse_lms), ("RLS (遞迴二階矩陣)", mse_rls), ("Adam (AI 動態步長)", mse_adam)]
    for algo, mse_data in algos_to_test:
        steady_state_mse = np.mean(mse_data[-500:])  # 取最後 500 點穩態平均
        status = "PASS" if steady_state_mse < spec_limit else "FAIL"
        print(f"  演算法 [{algo}] -> 穩態 MSE: {steady_state_mse:.5f} | 規格上限: {spec_limit} | 測試結果: {status}")

        # ==========================================
    # 6. 高清晰度分欄數據視覺化 (4x2 畫布設計)
    # ==========================================
    # 建立 4 列 2 行的畫布，左欄合併顯示收斂曲線，右欄拆解顯示星座圖
    fig = plt.figure(figsize=(16, 12))
    
    # ----------------------------------------
    # 左側：學習收斂曲線 (跨越 4 列的合併子圖)
    # ----------------------------------------
    ax_curve = plt.subplot2grid((4, 2), (0, 0), rowspan=4, colspan=1)
    ax_curve.plot(mse_lms, label="Traditional LMS", color="blue", alpha=0.4, linewidth=1)
    ax_curve.plot(mse_rls, label="High-order RLS", color="darkgreen", alpha=0.5, linewidth=1)
    ax_curve.plot(mse_adam, label="Hand-written Adam (AI)", color="red", alpha=0.7, linewidth=1.2)
    ax_curve.axhline(y=spec_limit, color="black", linestyle="--", label=f"Spec Limit ({spec_limit})", linewidth=1.5)
    ax_curve.set_yscale("log")
    ax_curve.set_title("Equalizer Convergence Curve (Learning Process)", fontsize=12, fontweight='bold')
    ax_curve.set_xlabel("Symbols", fontsize=10)
    ax_curve.set_ylabel("MSE (Log Scale)", fontsize=10)
    ax_curve.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_curve.legend(fontsize=10, loc="upper right")

    # ----------------------------------------
    # 右側：完全拆解的星座圖 (分開 4 個子圖獨立對比)
    # ----------------------------------------
    # 為了方便對比，統一所有星座圖的座標軸範圍
    axis_lim = [-1.5, 1.5, -1.5, 1.5]
    scatter_size = 6
    
    # 1. 接收端未等化訊號 (Rx Before Equalization)
    ax_rx = plt.subplot2grid((4, 2), (0, 1))
    ax_rx.scatter(rx_symbols[-500:].real, rx_symbols[-500:].imag, color="gray", alpha=0.5, s=scatter_size)
    ax_rx.set_title("1. Received Signal (Before Equalization)", fontsize=10, color="dimgray")
    ax_rx.axis(axis_lim)
    ax_rx.grid(True, linestyle=":", alpha=0.5)

    # 2. LMS 等化後訊號
    ax_lms = plt.subplot2grid((4, 2), (1, 1))
    ax_lms.scatter(y_out_lms[-500:].real, y_out_lms[-500:].imag, color="blue", alpha=0.6, s=scatter_size)
    ax_lms.set_title("2. After Traditional LMS (O(M))", fontsize=10, color="blue")
    ax_lms.axis(axis_lim)
    ax_lms.grid(True, linestyle=":", alpha=0.5)

    # 3. RLS 等化後訊號
    ax_rls = plt.subplot2grid((4, 2), (2, 1))
    ax_rls.scatter(y_out_rls[-500:].real, y_out_rls[-500:].imag, color="darkgreen", alpha=0.6, s=scatter_size)
    ax_rls.set_title("3. After High-order RLS (O(M²))", fontsize=10, color="darkgreen")
    ax_rls.axis(axis_lim)
    ax_rls.grid(True, linestyle=":", alpha=0.5)

    # 4. Adam 等化後訊號
    ax_adam = plt.subplot2grid((4, 2), (3, 1))
    ax_adam.scatter(y_out_adam[-500:].real, y_out_adam[-500:].imag, color="red", alpha=0.7, s=scatter_size)
    ax_adam.set_title("4. After Hand-written Adam (AI - O(M))", fontsize=10, color="red")
    ax_adam.axis(axis_lim)
    ax_adam.grid(True, linestyle=":", alpha=0.5)
    ax_adam.set_xlabel("In-Phase (I)", fontsize=9) 

    plt.tight_layout()
    plt.show()

    

if __name__ == "__main__":
    run_integrated_qa_test()

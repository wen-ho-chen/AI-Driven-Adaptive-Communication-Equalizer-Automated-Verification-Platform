# src/equalizer_dut.py
import numpy as np

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



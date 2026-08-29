# Baseband Channel Equalization Testbench (LMS/RLS)

這個專案使用 **Python** 實現了數位訊號處理（DSP）中經典的 **LMS (Least Mean Squares)** 與 **RLS (Recursive Least Squares)** 自適應濾波器演算法，並針對多徑干擾（ISI）與高斯白雜訊（AWGN）環境建立了一個自動化驗證測試平台（Testbench）。

## 🚀 核心功能與測試架構
1. **訊號源與通道模擬**：生成 QPSK 基頻訊號，並透過卷積（Convolution）模擬多徑衰落通道，動態注入 AWGN 雜訊。
2. **被測系統 (DUT)**：實現穩健的複數矩陣運算，修復了複數共軛轉置在 RLS 矩陣更新時的邊界問題。
3. **自動化驗證機制 (QA Verification)**：
   * 以穩態**均方誤差 (MSE)** 與 **EVM** 作為測試合格指標（Threshold Assertion）。
   * 設計不同 SNR 環境（如 25dB 正常環境與 5dB 惡劣環境）的壓力測試場景（Stress / Corner Case Test）。

## 🛠️ 開發環境
* Python 3.x
* NumPy (矩陣與複數運算)
* Matplotlib (波形與星座圖分析)

## 📊 測試結果
執行 `python main.py` 後，系統將自動輸出自動化測試日誌（Pass/Fail 判定），並繪製 QPSK 星座圖收斂對比。

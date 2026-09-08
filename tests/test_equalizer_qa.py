# tests/test_equalizer_qa.py
import pytest
import numpy as np
from src.equalizer_dut import AdaptiveEqualizerDUT

@pytest.fixture
def test_data():
    """完美還原會過的環境數據"""
    np.random.seed(42)
    num_symbols = 2000
    fading_channel = np.array([1.0, 0.4 - 0.3j, 0.1 + 0.2j])

    bits = np.random.randint(0, 2, num_symbols * 2)
    qpsk_mapping = {(0, 0): 1+1j, (0, 1): -1+1j, (1, 1): -1-1j, (1, 0): 1-1j}
    tx_symbols = np.array([qpsk_mapping[(bits[i], bits[i+1])] for i in range(0, len(bits), 2)]) / np.sqrt(2)

    snr_db = 15
    rx_faded = np.convolve(tx_symbols, fading_channel, mode="same")
    sig_power = np.mean(np.abs(rx_faded) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols))
    rx_symbols = rx_faded + noise

    return rx_symbols, tx_symbols

@pytest.mark.parametrize("algo_name", ["LMS", "RLS", "Adam"])
def test_equalizer_regression(test_data, algo_name):
    rx_symbols, tx_symbols = test_data
    
    # 🌟 核心關鍵：嚴格指定 filter_order=5, lr=0.01，與你原本會過的一模一樣！
    equalizer = AdaptiveEqualizerDUT(filter_order=5, lr=0.01)
    
    _, mse_data = equalizer.equalize(rx_symbols, tx_symbols, algo=algo_name)
    
    # 取最後 500 點穩態平均
    steady_state_mse = np.mean(mse_data[-500:])
    spec_limit = 0.1
    
    print(f"\n[Pytest CI] {algo_name} -> 穩態 MSE: {steady_state_mse:.5f}")
    
    # 自動化斷言
    assert steady_state_mse < spec_limit, f"{algo_name} 穩態效能未達標！"

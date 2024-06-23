import numpy as np
import matplotlib.pyplot as plt

def lms_filter(x, d, mu, filter_order):
    """
    LMS濾波器實現
    
    x: 輸入信號（包含噪音）
    d: 期望信號
    mu: 學習率
    filter_order: 濾波器階數
    
    返回：
    y: 濾波器輸出信號
    e: 誤差信號
    w: 最終濾波器權重
    """
    n = len(x)
    y = np.zeros(n)
    e = np.zeros(n)
    w = np.zeros(filter_order)
    
    for i in range(filter_order, n):
        x_vec = x[i:i-filter_order:-1]
        y[i] = np.dot(w, x_vec)
        e[i] = d[i] - y[i]
        w = w + 2 * mu * e[i] * x_vec
    
    return y, e, w

def moving_average_filter(x, window_size):
    """
    平均濾波器實現
    
    x: 輸入信號（包含噪音）
    window_size: 窗口大小
    
    返回：
    y: 濾波器輸出信號
    """
    y = np.convolve(x, np.ones(window_size)/window_size, mode='valid')
    return np.concatenate((np.zeros(window_size-1), y))

def kalman_filter(x, Q, R):
    """
    簡單卡爾曼濾波器實現
    
    x: 輸入信號（包含噪音）
    Q: 過程噪聲協方差
    R: 測量噪聲協方差
    
    返回：
    y: 濾波器輸出信號
    """
    n = len(x)
    y = np.zeros(n)
    P = 1.0
    x_hat = 0.0
    
    for i in range(n):
        # 預測
        x_hat = x_hat
        P = P + Q
        
        # 更新
        K = P / (P + R)
        x_hat = x_hat + K * (x[i] - x_hat)
        P = (1 - K) * P
        
        y[i] = x_hat
    
    return y

# 模擬信號
np.random.seed(0)
n_samples = 500
s = np.sin(2 * np.pi * 0.01 * np.arange(n_samples))  # 期望信號（正弦波）

# 測試不同類型的噪音
noise_types = {
    'Gaussian Noise': 0.5 * np.random.randn(n_samples),
    'Pink Noise': (np.random.randn(n_samples).cumsum() / np.sqrt(n_samples)),
    'Environmental Noise': 0.5 * np.random.randn(n_samples) * np.sin(2 * np.pi * 0.01 * np.arange(n_samples) * np.random.rand())
}

mu = 0.01
filter_order = 10

plt.figure(figsize=(12, 15))

for i, (noise_type, noise) in enumerate(noise_types.items(), start=1):
    x = s + noise
    y, e, w = lms_filter(x, s, mu, filter_order)

    # 子圖：接收到的信號
    plt.subplot(len(noise_types), 3, (i-1)*3 + 1)
    plt.plot(x, label='Noisy Signal')
    plt.title(f'Received Signal with {noise_type}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # 子圖：濾波後的信號
    plt.subplot(len(noise_types), 3, (i-1)*3 + 2)
    plt.plot(y, label='Filtered Signal')
    plt.title(f'Filtered Signal with {noise_type}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # 子圖：疊加比較
    plt.subplot(len(noise_types), 3, (i-1)*3 + 3)
    plt.plot(x, label='Noisy Signal', linestyle='--')
    plt.plot(s, label='Original Signal')
    plt.plot(y, label='Filtered Signal', linestyle='-.')
    plt.title(f'Signal Comparison with {noise_type}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# 使用高斯噪音比較不同降噪方法
noise = 0.5 * np.random.randn(n_samples)  # 高斯噪音
x = s + noise

window_size = 5
Q = 1e-5
R = 0.1

# LMS濾波器
y_lms, e_lms, w_lms = lms_filter(x, s, mu, filter_order)

# 平均濾波器
y_ma = moving_average_filter(x, window_size)

# 卡爾曼濾波器
y_kalman = kalman_filter(x, Q, R)

plt.figure(figsize=(15, 12))

# 子圖：LMS濾波器
plt.subplot(3, 1, 1)
plt.plot(x, label='Noisy Signal', linestyle='--')
plt.plot(s, label='Original Signal')
plt.plot(y_lms, label='LMS Filtered Signal', linestyle='-.')
plt.title('LMS Filter')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# 子圖：平均濾波器
plt.subplot(3, 1, 2)
plt.plot(x, label='Noisy Signal', linestyle='--')
plt.plot(s, label='Original Signal')
plt.plot(y_ma, label='Moving Average Filtered Signal', linestyle='-.')
plt.title('Moving Average Filter')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# 子圖：卡爾曼濾波器
plt.subplot(3, 1, 3)
plt.plot(x, label='Noisy Signal', linestyle='--')
plt.plot(s, label='Original Signal')
plt.plot(y_kalman, label='Kalman Filtered Signal', linestyle='-.')
plt.title('Kalman Filter')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

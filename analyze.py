import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df5 = pd.read_csv("outputs/scan_N5_alpha0.8.csv")

# 1. Frequency ratio vs lambda
ratio = df5['omega_in'] / df5['omega_out']
plt.plot(df5['lambda'], ratio, 'o-')
plt.axhline(1.0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('λ')
plt.ylabel('ω_in / ω_out')
plt.title('N=5 Frequency Inversion')
plt.grid(alpha=0.3)
plt.show()

# 2. Identify where inversion begins
inversion_start = df5[df5['omega_in'] < df5['omega_out']]['lambda'].iloc[0]
print(f"Frequency inversion begins at λ = {inversion_start:.3f}")

# 3. Entanglement asymmetry
df5['lambda_diff'] = df5['lambda_in'] - df5['lambda_out']
plt.plot(df5['lambda'], df5['lambda_diff'], 's-', color='coral')
plt.xlabel('λ')
plt.ylabel('ΔΛ = Λ_in - Λ_out')
plt.title('Edge Asymmetry Growth')
plt.grid(alpha=0.3)
plt.show()

# 4. Residual derivative (sharpness of transitions)
df5['residual_gradient'] = np.gradient(df5['residual'], df5['lambda'])
plt.plot(df5['lambda'], df5['residual_gradient'], '-', color='purple')
plt.axvline(inversion_start, color='red', linestyle=':')
plt.xlabel('λ')
plt.ylabel('d(residual)/dλ')
plt.title('Transition Sharpness')
plt.grid(alpha=0.3)
plt.show()

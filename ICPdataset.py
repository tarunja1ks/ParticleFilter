
import numpy as np

# Generate circle points
angles = np.linspace(0, 2*np.pi, 50, endpoint=False)
source = np.column_stack([np.cos(angles), np.sin(angles)])

# True transformation
theta = np.pi/6  # 30 degrees
R_true = np.array([[np.cos(theta), -np.sin(theta)], 
                   [np.sin(theta), np.cos(theta)]])
t_true = np.array([0.5, 0.3])

# Apply transformation
target = (R_true @ source.T).T + t_true

# Add noise
target += np.random.normal(0, 0.02, target.shape)

# Save dataset
np.savez('icp_dataset.npz', source=source, target=target, R=R_true, t=t_true)

print("Dataset saved to icp_dataset.npz")
print(f"Source points: {source.shape}")
print(f"Target points: {target.shape}")
print(f"True rotation (degrees): {np.degrees(theta)}")
print(f"True translation: {t_true}")
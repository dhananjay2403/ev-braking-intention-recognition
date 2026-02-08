'''
    Multitask Data Generator 

    Previously (HARD dataset)

    Each sample had:
        - Input X → (75, 3)
        - Label y_class → {Light, Normal, Emergency}

    Now (Multitask HARD dataset)

    Each sample will have:
        - Input X → (75, 3)
        - Classification label y_class → {0,1,2}
        - Regression label y_intensity → continuous value in [0,1]

    y_intensity represents the future braking intensity the driver is heading toward.
'''

import numpy as np


'''
    Generates one ambiguous braking sequence for multitask learning.

    Returns:
        X           : (seq_len, 3)
        y_class     : int (0,1,2)
        y_intensity : float in [0,1] (future braking intensity)
'''

def generate_hard_sample_mtl(seq_len=75):
    
    # Initial conditions
    speed = np.random.uniform(30, 90)
    accel = 0.0
    brake = 0.0

    X = []

    # Decide FUTURE braking intensity
    future_intensity = np.random.rand()

    if future_intensity < 0.35:
        y_class = 0  # Light
        target_brake = np.random.uniform(0.15, 0.35)
    elif future_intensity < 0.7:
        y_class = 1  # Normal
        target_brake = np.random.uniform(0.30, 0.60)
    else:
        y_class = 2  # Emergency
        target_brake = np.random.uniform(0.55, 1.00)

    # Generate time-series
    for _ in range(seq_len):
        # Gradual braking ramp
        brake += (target_brake - brake) * np.random.uniform(0.02, 0.08)

        # Driver inconsistency & sensor noise
        brake += np.random.normal(0, 0.05)
        brake = np.clip(brake, 0, 1)

        # Acceleration responds imperfectly
        accel = -brake * np.random.uniform(1.5, 3.5)
        accel += np.random.normal(0, 0.2)

        # Speed update
        speed += accel * 0.1
        speed = max(speed, 0)

        X.append([speed, accel, brake])

    return np.array(X), y_class, target_brake


def generate_dataset_mtl(n_samples = 15000):
    X, y_class, y_intensity = [], [], []

    for _ in range(n_samples):
        x_i, yc_i, yi_i = generate_hard_sample_mtl()
        X.append(x_i)
        y_class.append(yc_i)
        y_intensity.append(yi_i)

    return (
        np.array(X),
        np.array(y_class),
        np.array(y_intensity)
    )


if __name__ == "__main__":

    np.random.seed(42)

    X, y_class, y_intensity = generate_dataset_mtl()

    # Shuffle
    idx = np.random.permutation(len(X))
    X = X[idx]
    y_class = y_class[idx]
    y_intensity = y_intensity[idx]

    # Split
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.85 * n)

    X_train, X_val, X_test = X[:n_train], X[n_train : n_val], X[n_val:]
    y_class_train, y_class_val, y_class_test = (
        y_class[:n_train],
        y_class[n_train : n_val],
        y_class[n_val:]
    )
    y_int_train, y_int_val, y_int_test = (
        y_intensity[:n_train],
        y_intensity[n_train : n_val],
        y_intensity[n_val:]
    )

    # Save
    np.save("data/X_train_hard_mtl.npy", X_train)
    np.save("data/X_val_hard_mtl.npy", X_val)
    np.save("data/X_test_hard_mtl.npy", X_test)

    np.save("data/y_class_train_hard_mtl.npy", y_class_train)
    np.save("data/y_class_val_hard_mtl.npy", y_class_val)
    np.save("data/y_class_test_hard_mtl.npy", y_class_test)

    np.save("data/y_int_train_hard_mtl.npy", y_int_train)
    np.save("data/y_int_val_hard_mtl.npy", y_int_val)
    np.save("data/y_int_test_hard_mtl.npy", y_int_test)

    print("Multitask HARD dataset generated.")




'''
    Data Generator

    We want to generate synthetic driving + braking behavior that looks realistic enough for ML.

    Each sample will represent:

    “The last 1.5 seconds of driving → what kind of braking is happening?”

    Signals we generate (same spirit as the Springer paper, but simpler):

    At every time step:
        - speed (m/s)
        - acceleration (m/s²)
        - brake_pedal (0 to 1)

    From these signals, we create:
        - Light braking
        - Normal braking
        - Emergency braking

    We then:
        - cut the continuous signals into sliding windows
        - assign one label per window
        - save everything as NumPy arrays (easy to load later)
'''


'''
    Dataset design recap (important, but short)

    - Sampling rate: 50 Hz
    - Window length: 1.5 seconds
    - Samples per window: 75
    - Features per timestep: 3
    - Output classes:
        0 → Light braking
        1 → Normal braking
        2 → Emergency braking

    Final ML input shape:

    X → (num_samples, 75, 3)
    y → (num_samples,)
'''


import numpy as np

# -------------------------------
# Global configuration
# -------------------------------

SAMPLING_RATE = 50        # Hz
WINDOW_SECONDS = 1.5
WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_SECONDS)  # 75

NUM_SAMPLES_PER_CLASS = 5000   # keep small first, can increase later
NOISE_STD_SPEED = 0.2          # m/s
NOISE_STD_BRAKE = 0.03

np.random.seed(42)  # reproducibility


# -------------------------------
# Helper functions
# -------------------------------

def add_noise(signal, std):
    """Add small Gaussian noise to make data realistic."""
    return signal + np.random.normal(0, std, size=signal.shape)


def generate_braking_event(brake_type):
    """
    Generate ONE braking event (continuous signal),
    from which we will extract one window.
    """

    t = np.linspace(0, WINDOW_SECONDS, WINDOW_SIZE)

    # Initial speed (random but realistic)
    v0 = np.random.uniform(15, 25)  # m/s (~54–90 km/h)

    if brake_type == "light":
        decel = np.random.uniform(0.5, 1.5)
        brake_level = np.random.uniform(0.05, 0.25)

    elif brake_type == "normal":
        decel = np.random.uniform(1.5, 3.5)
        brake_level = np.random.uniform(0.25, 0.6)

    elif brake_type == "emergency":
        decel = np.random.uniform(3.5, 6.0)
        brake_level = np.random.uniform(0.7, 1.0)

    else:
        raise ValueError("Unknown brake type")

    # Speed profile (simple linear deceleration)
    speed = v0 - decel * t
    speed = np.clip(speed, 0, None)

    # Acceleration (negative during braking)
    acceleration = -np.ones_like(t) * decel

    # Brake pedal profile (smooth ramp)
    brake_pedal = brake_level * (1 - np.exp(-5 * t))

    # Add noise
    speed = add_noise(speed, NOISE_STD_SPEED)
    brake_pedal = add_noise(brake_pedal, NOISE_STD_BRAKE)

    # Stack features: (time, features)
    sample = np.stack([speed, acceleration, brake_pedal], axis=1)

    return sample


# -------------------------------
# Dataset generation
# -------------------------------

def generate_dataset():
    X = []
    y = []

    classes = {
        "light": 0,
        "normal": 1,
        "emergency": 2
    }

    for brake_type, label in classes.items():
        print(f"Generating {brake_type} braking samples...")

        for _ in range(NUM_SAMPLES_PER_CLASS):
            sample = generate_braking_event(brake_type)
            X.append(sample)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y


# -------------------------------
# Train / Val / Test split
# -------------------------------

def split_dataset(X, y, train_ratio=0.7, val_ratio=0.15):
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    n_train = int(train_ratio * len(X))
    n_val = int(val_ratio * len(X))

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]

    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test


# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    print("Generating dataset...")
    X, y = generate_dataset()

    print("Splitting dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)

    print("Saving files...")
    np.save("data/X_train.npy", X_train)
    np.save("data/y_train.npy", y_train)
    np.save("data/X_val.npy", X_val)
    np.save("data/y_val.npy", y_val)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_test.npy", y_test)

    print("Done!")
    print("Shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    cols = ['type', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    df = pd.read_csv(filename, skiprows=3, header=None, names=cols)
    return df

def compute_phase_space(df):
    r = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    v = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
    return r, v

def plot_phase(df, galaxy_type, galaxy_name, velocity_label):
    df_halo = df[df['type'] == galaxy_type]
    r, v = compute_phase_space(df_halo)

    plt.figure(figsize=(8,6))
    plt.scatter(r, v, s=1, alpha=0.5)
    plt.xlabel("Radius [kpc]")
    plt.ylabel("Velocity [km/s]")
    plt.title(f"{galaxy_name} Phase Space ({velocity_label})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"phase_space_{galaxy_name.lower().replace(' ', '_')}_{velocity_label}.png")
    plt.close()

# Example usage
if __name__ == "__main__":
    files = {
        "100kms": "1ascii.csv",
        "200kms": "11ascii.csv",
        "300kms": "3ascii.csv"
    }

    for velocity, file in files.items():
        df = load_data(file)
        plot_phase(df, 0, "Galaxy 1", velocity)
        plot_phase(df, 3, "Galaxy 2", velocity)

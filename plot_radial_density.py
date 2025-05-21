import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    # Skip the first 3 rows (metadata) and set proper column names
    cols = ['type', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    df = pd.read_csv(filename, skiprows=3, header=None, names=cols)
    return df

def compute_radius(df):
    return np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

def plot_radial_density(df, galaxy_type, galaxy_name, velocity_label):
    # Filter for specific galaxy halo type (0 for Galaxy 1, 3 for Galaxy 2)
    df_halo = df[df['type'] == galaxy_type]
    r = compute_radius(df_halo)

    # Bin and normalize
    counts, bins = np.histogram(r, bins=50, range=(0, 300))
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    density = counts / (4/3 * np.pi * (bins[1:]**3 - bins[:-1]**3))

    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(bin_centers, density)
    plt.xlabel("Radius [kpc]")
    plt.ylabel("Density [particles/kpc³]")
    plt.title(f"{galaxy_name} Radial Density Profile ({velocity_label})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"radial_density_{galaxy_name.lower().replace(' ', '_')}_{velocity_label}.png")
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
        plot_radial_density(df, 0, "Galaxy 1", velocity)
        plot_radial_density(df, 3, "Galaxy 2", velocity)

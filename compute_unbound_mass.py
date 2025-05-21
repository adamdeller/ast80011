import pandas as pd
import numpy as np

def load_data(filename):
    cols = ['type', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    df = pd.read_csv(filename, skiprows=3, header=None, names=cols)
    return df

def compute_unbound_fraction(df, galaxy_type, softening_length=1.0):
    df_halo = df[df['type'] == galaxy_type]

    # Position and velocity magnitudes
    r = np.sqrt(df_halo['x']**2 + df_halo['y']**2 + df_halo['z']**2) + softening_length
    v2 = df_halo['vx']**2 + df_halo['vy']**2 + df_halo['vz']**2

    # Approximate potential energy assuming mass = 1 (since mass is not given)
    G = 4.302e-6  # kpc * (km/s)^2 / Msun
    M = len(df_halo)  # number of particles, assume unit mass
    E_kin = 0.5 * v2
    E_pot = -G * M / r
    E_total = E_kin + E_pot

    unbound_fraction = np.sum(E_total > 0) / len(df_halo)
    return unbound_fraction

# Example usage
if __name__ == "__main__":
    files = {
        "100kms": "1ascii.csv",
        "200kms": "11ascii.csv",
        "300kms": "3ascii.csv"
    }

    results = []
    for velocity, file in files.items():
        df = load_data(file)
        f1 = compute_unbound_fraction(df, 0)
        f2 = compute_unbound_fraction(df, 3)
        results.append([velocity, f1, f2])

    df_results = pd.DataFrame(results, columns=["Velocity", "Galaxy 1 Unbound Fraction", "Galaxy 2 Unbound Fraction"])
    df_results.to_csv("unbound_mass_fractions.csv", index=False)
    print(df_results)

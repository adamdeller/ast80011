import pandas as pd
import numpy as np

def load_data(filename):
    cols = ['type', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    df = pd.read_csv(filename, skiprows=3, header=None, names=cols)
    return df

def compute_stats(df, galaxy_type):
    df_halo = df[df['type'] == galaxy_type]

    # Radius and velocity magnitude
    r = np.sqrt(df_halo['x']**2 + df_halo['y']**2 + df_halo['z']**2)
    v = np.sqrt(df_halo['vx']**2 + df_halo['vy']**2 + df_halo['vz']**2)

    # Central density: particles within 10 kpc
    central_density = np.sum(r < 10) / ((4/3) * np.pi * 10**3)

    mean_r = np.mean(r)
    mean_v = np.mean(v)
    vel_disp = np.std(v)

    return mean_r, central_density, mean_v, vel_disp

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
        g1_stats = compute_stats(df, 0)
        g2_stats = compute_stats(df, 3)
        results.append([
            velocity,
            *g1_stats,
            *g2_stats
        ])

    columns = [
        "Velocity",
        "G1 Mean Radius", "G1 Central Density", "G1 Mean Velocity", "G1 Velocity Dispersion",
        "G2 Mean Radius", "G2 Central Density", "G2 Mean Velocity", "G2 Velocity Dispersion"
    ]

    df_results = pd.DataFrame(results, columns=columns)
    df_results.to_csv("halo_stats_summary.csv", index=False)
    print(df_results)

#!/usr/bin/env python
"""
Script to plot the positions of planets and test particles at the end of the simulation in 3d
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import sys

if __name__ == "__main__":
    # Check for correct invocation
    if not len(sys.argv) == 2:
        print("Usage: {0} <csv file>".format(sys.argv[0]))
        sys.exit()

    # Load in the csv data into a pandas dataframe
    alldata = pd.read_csv(sys.argv[1], skipinitialspace=True)

    # Extract just the info corresponding to the last timestep and split into planets and test particles
    lasttimestep = alldata[alldata['time[yr]'] == alldata.iloc[-1]['time[yr]']]
    planets = lasttimestep[lasttimestep['type[tp/pl]'] == 'pl']
    testparticles = lasttimestep[lasttimestep['type[tp/pl]'] == 'tp']

    # Create a figure to plot on
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the planets in one colour
    print(lasttimestep['type[tp/pl]'])
    ax.scatter(planets['x[AU]'], planets['y[AU]'], planets['z[AU]'], marker='o', color='r')

    # Plot the test particles in another colour
    ax.scatter(testparticles['x[AU]'], testparticles['y[AU]'], testparticles['z[AU]'], marker='x', color='k')

    # Label the axes
    ax.set_xlabel('X position (AU)')
    ax.set_ylabel('Y position (AU)')
    ax.set_zlabel('Z position (AU)')

    # Save the figure
    fig.savefig("sd-3d.png")

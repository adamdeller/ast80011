#!/usr/bin/env python
#-------------------------------------------------------------------------------
# Name:        daisyChain.py
# Purpose:     Grabs the final positions of particles and planets from a run 
#              of the SD module and sets up the necessary input files to 
#              continue the simulation where it left off. Enables you to run
#              longer simulations.
#
# Created:     2021 by Adam Deller
#
#              This program is free software: you can redistribute it and/or modify it under
#              the terms of the GNU General Public License as published by the Free Software
#              Foundation, either version 3 of the License, or (at your option) any later
#              version.
#-------------------------------------------------------------------------------
import pandas as pd
import argparse, os
import json, math
from pandas import read_csv

# Run if desired
if __name__ == "__main__":
    # Get the paths to the input and output files
    parser = argparse.ArgumentParser(description='Convert the final entry in an AST80011 SD module output CSV file to planet and TP input CSV file')
    parser.add_argument('--previousjson', default='params.json',  help='Path to the json file that ran the simulation to be continued')
    parser.add_argument('--outputcsv', default="ascii.csv", type=str, help='Path to the (unzipped) ascii.csv file from a previous SD module run')
    parser.add_argument('--newplcsv', default="daisychainplanets.csv", help="Path to planets.csv file for a new SD module run")
    parser.add_argument('--newtpcsv', default="daisychaintestparticles.csv", help="Path to testparticles.csv file for a new SD module run")
    parser.add_argument('--newjson', default="daisychain.json",  help="Path to params.json file for a new SD module run")
    args = parser.parse_args()

    # Check that the to-be-translated csv file exists
    if not os.path.exists(args.outputcsv):
        parser.error(args.outputcsv + " doesn't exist")

    # Check that the json file from the simulation to be continued exists
    if not os.path.exists(args.previousjson):
        parser.error(args.previousjson + " doesn't exist")

    # Parse the json file
    with open(args.previousjson) as jsonin:
        params = json.load(jsonin)

    # Read in the output from the previous SD run
    df = read_csv(args.outputcsv,skipinitialspace=True)

    # Select the final timestep
    dfout = df[df["time[yr]"] == df.iloc[-1]["time[yr]"]].copy()

    # Select out the planets and test particles separately
    pldf = dfout[dfout["type[tp/pl]"] == "pl"].copy()
    tpdf = dfout[dfout["type[tp/pl]"] == "tp"].copy()

    # At this point, we have two dataframes in the format:
    # time[yr],type[tp/pl],particle no.,x[AU],y[AU],z[AU],vx[AU/year],vy[AU/year],vz[AU/year],a,e,inc[degrees]
    # and we want to change the order to 
    # eccentricity, semi-major axis, inclination[, mass], x, y, z, vx, vy, vz[, close encounter radius] (the last is for the planets only)

    # Walk through the json file and grab the planet masses and close encounter radii
    masssum = 0
    masses = []
    closeradii = []
    for i, p in enumerate(params['planets']):
        masses.append(p['mass'])
        closeradii.append(p['close'])
        masssum += masses[-1]
        # Double check that the index of this planet matches expectations
        if not p['index'] == i:
            raise Exception("Planet index",  p['index'], " of the ", i, "th planet in the json file doesn't match expectations! Aborting.")
    #print(pldf)
    #print(masses)
    pldf['mass'] = masses
    pldf['closeradii'] = closeradii
    
    # Re-order the columns in the planets dataframe 
    plcols = ["e","a","inc[degrees]","mass","x[AU]","y[AU]","z[AU]","vx[AU/year]","vy[AU/year]","vz[AU/year]","closeradii"]
    pldf = pldf[plcols]

    # Re-order the columns in the test particle dataframe
    tpcols = ["e","a","inc[degrees]","x[AU]","y[AU]","z[AU]","vx[AU/year]","vy[AU/year]","vz[AU/year]"]
    tpdf = tpdf[tpcols]

    # Scale all the velocities by default_central_mass / (central_mass + planet mass sum), plus a small fudge factor
    # Needed to convert from AU/yr into SWIFT velocity units (although I don't understand the small fudge factor)
    velocitycolumns = ["vx[AU/year]","vy[AU/year]","vz[AU/year]"]
    default_central_mass = 1047 # Jupiter masses
    fudge_factor = 1.00115091
    for index, row in pldf.iterrows():
        for v in velocitycolumns:
            row[v] /= fudge_factor * math.sqrt(default_central_mass / (params['cent_mass']  + row['mass']))
    for v in velocitycolumns:
        #pldf[v] = pldf[v] / (fudge_factor * math.sqrt(default_central_mass / (params['cent_mass']  + pldf['mass'])))
        tpdf[v] = tpdf[v] / (fudge_factor * math.sqrt(default_central_mass / params['cent_mass']))

    # Filter out any test particles that were lost
    livetpdf = tpdf.loc[(tpdf[velocitycolumns]!=0).any(axis=1)]

    # Write out the two csv files
    pldf.to_csv(args.newplcsv, header=False, index=False)
    livetpdf.to_csv(args.newtpcsv, header=False, index=False)

    # Also write out a whole new params.json file ready to go
    for i, p in enumerate(params['planets']):
        p['ecc'] = pldf.iloc[i]["e"]
        p['sma'] = pldf.iloc[i]["a"]
        p['inc'] = pldf.iloc[i]["inc[degrees]"]
        p['x'] = pldf.iloc[i]["x[AU]"]
        p['y'] = pldf.iloc[i]["y[AU]"]
        p['z'] = pldf.iloc[i]["z[AU]"]
        p['vx'] = pldf.iloc[i]["vx[AU/year]"]
        p['vy'] = pldf.iloc[i]["vy[AU/year]"]
        p['vz'] = pldf.iloc[i]["vz[AU/year]"]
    # Ensure that the json file now uses fully specified test particle info rather than randomise, 
    # even if the simulation that is being continued used randomisation initially.
    params['randomize_particles'] = False
    params['num_particles'] = livetpdf.shape[0]
    params['particles'] = []
    for i in range(params['num_particles']):
        t = {}
        t['index'] = i
        t['ecc'] = livetpdf.iloc[i]["e"]
        t['sma'] = livetpdf.iloc[i]["a"]
        t['inc'] = livetpdf.iloc[i]["inc[degrees]"]
        t['x'] = livetpdf.iloc[i]["x[AU]"]
        t['y'] = livetpdf.iloc[i]["y[AU]"]
        t['z'] = livetpdf.iloc[i]["z[AU]"]
        t['vx'] = livetpdf.iloc[i]["vx[AU/year]"]
        t['vy'] = livetpdf.iloc[i]["vy[AU/year]"]
        t['vz'] = livetpdf.iloc[i]["vz[AU/year]"]
        params['particles'].append(t)
    #for i, t in enumerate(params['particles']):
    #    t['ecc'] = tpdf.iloc[i]["e"]
    #    t['sma'] = tpdf.iloc[i]["a"]
    #    t['inc'] = tpdf.iloc[i]["inc[degrees]"]
    #    t['x'] = tpdf.iloc[i]["x[AU]"]
    #    t['y'] = tpdf.iloc[i]["y[AU]"]
    #    t['z'] = tpdf.iloc[i]["z[AU]"]
    #    t['vx'] = tpdf.iloc[i]["vx[AU/year]"]
    #    t['vy'] = tpdf.iloc[i]["vy[AU/year]"]
    #    t['vz'] = tpdf.iloc[i]["vz[AU/year]"]
    with open(args.newjson,"w") as jsonout:
        json.dump(params,jsonout)

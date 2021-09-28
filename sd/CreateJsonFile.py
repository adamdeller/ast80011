#-------------------------------------------------------------------------------
# Name:        CreateJsonFile
# Purpose:     Updates the (x,y,z) and (vx,yx,zx) of a pre-created param.json
#              (an output from a sim run of the solar system) to the values in
#              ephemeris file DE440.bsp. It then adds a rogue planet based on
#              my calculations
#
# Notes:       The Julian Day must be set to whatever day required
#              If the rogue planet is not desired set ROGUE to False
# Author:      Hans van der Heyden
#
# Created:     26/09/2021
# Copyright:   (c) Hans van der Heyden 2021
# Licence:     None. Use at your own risk.
#-------------------------------------------------------------------------------
from jplephem.spk import SPK
import numpy as np
import json

# Convertion factors
AU = 6.684587E-9    # convert km to AU
AUYR = 2.44155E-6   # convert km/s to AU/Yr

# the required JD change as necessary
JD = 2414864.5      # start julian day

# if the rogue planet is desired set to True
ROGUE = True

# position and velocity data for each solar system planet at start in the following order
#names = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto', 'Sun' 'Planet']
positions = []
velocities = []

def main():
    # Open the ephemeris file
    kernel = SPK.open('de440.bsp')

    # extract the position data
    for i in range(1,11):
        p, v = kernel[0,i].compute_and_differentiate(JD)
        positions.append(p)
        velocities.append(v)

    # now open the json file and add above values
    js = open('params.json')
    jsData = json.load(js)
    js.close()

    jsPlanets = jsData['planets']

    # Translate the files and store them in the JSON File
    for p in range(9):
        jsPlanets[p]['x'] = (positions[p][0] - positions[9][0])*AU
        jsPlanets[p]['y'] = (positions[p][1] - positions[9][1])*AU
        jsPlanets[p]['z'] = (positions[p][2] - positions[9][2])*AU
        jsPlanets[p]['vx'] = velocities[p][0] * AUYR
        jsPlanets[p]['vy'] = velocities[p][1] * AUYR
        jsPlanets[p]['vz'] = velocities[p][2] * AUYR
        rp = p

    if ROGUE:
        # add the rogue planet
        rgPlanet = { "mass": 1,
                     "close": 0.35529968,
                     "index": rp+1,
                     "ecc": 0.9947,
                     "sma": 3.5142e10*AU,
                     "inc": 90.0,
                     "x": -39.539,
                     "y": 1.1243,
                     "z": -13.6138,
                     "vx":-8.6311e-2,
                     "vy": 0.0,
                     "vz": 0.2509}

        jsPlanets.append(rgPlanet)

    # write the file
    outjs = open('pararms2.json','w')
    json.dump(jsData,outjs)
    outjs.close()

if __name__ == '__main__':
    np.set_printoptions(precision=4)
    main()

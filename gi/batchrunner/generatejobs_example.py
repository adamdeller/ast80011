#!.venv/bin/python3

# Example script to generate job input files for the GI batch runner script.
# Iterates for some relative vertical position angles for Galaxy 2 relative to Galaxy 1.
#    Copyright (C) 2020  DavidWh.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import math
import os
import json


def write_json_file(file_name, json_data):
    # Write a dictionary to disk as JSON
    with open(file_name, "w", encoding='utf-8') as json_file:
        json.dump(json_data, json_file)


def generate_jobs():
    # Set up default parameters for Galaxy 1
    galaxy1 = {
        "galaxy_number": 1,
        "components_to_use": ["disk", "bulge", "halo"],
        "bulge": {
            "number_of_particles": 500
        },
        "disk": {
            "number_of_particles": 500,
            "mass_factor": 0.867
        },
        "halo": {
            "number_of_particles": 500,
            "potential_well_depth": -4.6
        }
    }

    # Set up default parameters for Galaxy 2
    galaxy2 = {
        "galaxy_number": 2,
        "components_to_use": ["bulge", "halo"],
        "bulge": {
            "number_of_particles": 500
        },
        "halo": {
            "number_of_particles": 500,
            "potential_well_depth": -4.6
        }
    }

    # Set up default parameters for Merge operation
    merge = {
        "length_scale": {
            "galaxy_1": 1,
            "galaxy_2": 0.4
        },
        "position_galaxy_2": {
            "x": 100,
            "y": 100,
            "z": 0
        },
        "velocity_galaxy_2": {
            "vx": -100,
            "vy": -100,
            "vz": 0
        },
        "theta_galaxy_2": 45,
    }

    # Set up default parameters for the Interaction
    interact = {
        "duration": 6.0,
        "nFrames": 100,
        "softening": 1,
        "mass_1": 15,
        "mass_2": 1.5,
    }

    # Set up default parameters for Plot 1
    plot1 = {
        "x_axis": "x",
        "y_axis": "z",
        "x_min": -500,
        "x_max": 500,
        "y_min": -500,
        "y_max": 500,
        "colour_scheme": "whiteonblack",
        "plot_size": "large"
    }

    # Set up default parameters for Plot 2
    plot2 = {
        "x_axis": "x",
        "y_axis": "y",
        "x_min": -500,
        "x_max": 500,
        "y_min": -500,
        "y_max": 500,
        "colour_scheme": "whiteonblack",
        "plot_size": "large"
    }

    # Set up default parameters for Plot 3
    plot3 = {
        "x_axis": "y",
        "y_axis": "z",
        "x_min": -500,
        "x_max": 500,
        "y_min": -500,
        "y_max": 500,
        "colour_scheme": "whiteonblack",
        "plot_size": "large"
    }

    # Path to output all the job subfolders to.
    JOBS_PATH = './Jobs'

    # Keep track of the output job number.
    job_number = 1

    # Required Distance and Speed of Galaxy 2 used to calculate the position and velocity vectors.
    galaxy2_distance = 200
    galaxy2_speed = 100

    # Iterate through vertical position angles of Galaxy 2 with respect to Galaxy 1, of between 0 and 90 degrees.
    horizontal_pos_angle = 0
    vertical_pos_angle = 0
    while vertical_pos_angle <= 90:
        # Calculate the position vector for Galaxy 2
        pos_galaxy_2_x = galaxy2_distance * \
            math.sin(math.radians(vertical_pos_angle)) * math.cos(math.radians(horizontal_pos_angle))
        pos_galaxy_2_y = galaxy2_distance * \
            math.sin(math.radians(vertical_pos_angle)) * math.sin(math.radians(horizontal_pos_angle))
        pos_galaxy_2_z = galaxy2_distance * math.cos(math.radians(vertical_pos_angle))
        merge["position_galaxy_2"]["x"] = pos_galaxy_2_x
        merge["position_galaxy_2"]["y"] = pos_galaxy_2_y
        merge["position_galaxy_2"]["z"] = pos_galaxy_2_z

        # Calculate the velocity vector for Galaxy 2. The velocity vector will be
        # pointed directly 'at' the centre of Galaxy 1.
        horizontal_velocity_angle = horizontal_pos_angle + 180
        vertical_velocity_angle = vertical_pos_angle + 180

        velocity_galaxy_2_x = galaxy2_speed * \
            math.sin(math.radians(vertical_velocity_angle)) * math.cos(math.radians(horizontal_velocity_angle))
        velocity_galaxy_2_y = galaxy2_speed * \
            math.sin(math.radians(vertical_velocity_angle)) * math.sin(math.radians(horizontal_velocity_angle))
        velocity_galaxy_2_z = galaxy2_speed * math.cos(math.radians(vertical_velocity_angle))
        merge["velocity_galaxy_2"]["x"] = velocity_galaxy_2_x
        merge["velocity_galaxy_2"]["y"] = velocity_galaxy_2_y
        merge["velocity_galaxy_2"]["z"] = velocity_galaxy_2_z

        # Create the output subfolder directory.
        job_dir = os.path.join(JOBS_PATH, str(job_number))
        os.makedirs(job_dir, exist_ok=True)

        # Write all output files.
        write_json_file(os.path.join(job_dir, 'galaxy1.json'), galaxy1)
        write_json_file(os.path.join(job_dir, 'galaxy2.json'), galaxy2)
        write_json_file(os.path.join(job_dir, 'merge.json'), merge)
        write_json_file(os.path.join(job_dir, 'interact.json'), interact)
        write_json_file(os.path.join(job_dir, 'plot1.json'), plot1)
        write_json_file(os.path.join(job_dir, 'plot2.json'), plot2)
        write_json_file(os.path.join(job_dir, 'plot3.json'), plot3)

        # Increment the job number.
        job_number += 1

        # Increment the vertical position angle.
        vertical_pos_angle += 30


generate_jobs()

#### ------------------- Importing modules ------------------- ####

import pandas as pd
import os
import gzip
import shutil
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import numpy as np
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
import matplotlib
from IPython.display import HTML
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import colors
import seaborn as sns
from scipy.optimize import curve_fit, least_squares
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import ast

#Hide warnings
import warnings

#### ------------------- Constants ------------------- ####
file_path_start = '/Users/a/Downloads/ast80011-main/gi/batchrunner/jobs_back/'
file_path_end = '/output/Galaxy Interaction'
start_timestep = 4
end_timestep = 6
#Find the current directory
current_dir = os.getcwd()

#Load extra_functions.py in the current_dir
sys.path.append(current_dir)

#### ------------------- Cleaning Data ------------------- ####

def read_data(file_num):
    ###     Description         ###
    #    This function takes a file number and reads 
    #       the data from the file. It then cleans it
    #    It returns a pandas dataframe
    ###

    # define the file path
    file_path = '/Users/a/Downloads/ast80011-main/gi/batchrunner/jobs_back/'+str(file_num)+'/output/Galaxy Interaction/_output_ascii.csv'


    if os.path.exists(file_path):
        print('File exists')
        # load the file into a pandas dataframe, skipping the first two rows
        df = pd.read_csv(file_path, skiprows=2)
    else:
        #add gz to file path
        file_pathgz = file_path + '.gz'
        #print the file_pathgz to the debug console
        print(file_pathgz)
        
        with gzip.open(file_pathgz, 'r') as fh:
            try: #This part opens it while the file part is still as .gz
                fh.read(1)
                #read file into df
                df = pd.read_csv(fh, skiprows=2)
            except gzip.BadGzipFile:
                print('input_file is not a valid gzip file by BadGzipFile')
        #print('Error in unzipping the file or file does not exist - this happens sometimes')
        
    # get the title from the third row
    title = df.columns[0]
    df.reset_index()

    df['timestep_val'] = df[' particles type'].where(df[' particles type'].str.contains('TIMESTEP')).ffill()  # extract the timestep value into a new column
    df = df.loc[~df[' particles type'].str.contains('TIMESTEP')]  # remove rows with TIMESTEP values
    #Remove the text 'TIMESTEP = ' from the timestep column
    df['timestep_val'] = df['timestep_val'].str.replace('TIMESTEP = ', '')
    #turn timestep column into a float
    df['timestep_val'] = df['timestep_val'].astype(float)
    #print(df.head())
    return df


def adjust_data(df):

    ###     Description         ###
    #    This function takes a dataframe and adjusts 
    #    the x y and z values and vx vy vz 
    #    so that they are relative 
    #    to the centre of mass of the system at each timestep
    ###

    #As particles often fly off in random directions in the simulation 
    # we first want to remove these else they impact the averages and 
    # the whole analysis. 
    # Therefore we remove any datapoints more than 3sd out per particle type and per timestep
    #We do this for the position first and group by the values per timestep and particle type. Do this in one line
    
    # Define columns to check for outliers
    position_velocity_columns = [' x[kpc]', ' y[kpc]', ' z[kpc]']#, 'vx[km/s]', ' vy[km/s]', ' vz[km/s]']

    # Function to remove outliers
    def remove_outliers(group):
        for col in position_velocity_columns:
            group = group[np.abs(group[col] - group[col].mean()) <= (5*group[col].std())]
        return group
    
    # Apply the function to each group
    df = df.groupby(['timestep_val', ' particles type']).apply(remove_outliers)

    # Reset index after groupby operation
    df.reset_index(drop=True, inplace=True)

    #Add a column which is the mean of the x y and z values for each timestep and particle type
    df['mean_x'] = df.groupby(['timestep_val', ' particles type'])[' x[kpc]'].transform('mean')
    df['mean_y'] = df.groupby(['timestep_val', ' particles type'])[' y[kpc]'].transform('mean')
    df['mean_z'] = df.groupby(['timestep_val', ' particles type'])[' z[kpc]'].transform('mean')

    #Add a column which is the mean of the vx vy and vz values for each timestep and particle type
    df['mean_vx'] = df.groupby(['timestep_val', ' particles type'])['vx[km/s]'].transform('mean')
    df['mean_vy'] = df.groupby(['timestep_val', ' particles type'])[' vy[km/s]'].transform('mean')
    df['mean_vz'] = df.groupby(['timestep_val', ' particles type'])[' vz[km/s]'].transform('mean')

    #Add a backup of the x y and z values and vx vy and vz values
    df['x_raw'] = df[' x[kpc]']
    df['y_raw'] = df[' y[kpc]']
    df['z_raw'] = df[' z[kpc]']
    df['vx_raw'] = df['vx[km/s]']
    df['vy_raw'] = df[' vy[km/s]']
    df['vz_raw'] = df[' vz[km/s]']

    #Subtract the mean of the x y and z values from the x y and z values
    df[' x[kpc]'] = df[' x[kpc]'] - df['mean_x']
    df[' y[kpc]'] = df[' y[kpc]'] - df['mean_y']
    df[' z[kpc]'] = df[' z[kpc]'] - df['mean_z']

    #Subtract the mean of the vx vy and vz values from the vx vy and vz values
    df['vx[km/s]'] = df['vx[km/s]'] - df['mean_vx']
    df[' vy[km/s]'] = df[' vy[km/s]'] - df['mean_vy']
    df[' vz[km/s]'] = df[' vz[km/s]'] - df['mean_vz']

    #return the df
    return df


#Create a function which will add a column the length of the vector to the DataFrame
def add_vector_length(df):
    ####   Description     ####
    # This function adds a column to the dataframe
    # which is the length of the vector of the x y and z values
    # Input: df - dataframe of the simulation data
    # Output: df - dataframe of the simulation data with a column of the length of the vector
    df['len'] = np.sqrt(df[' x[kpc]']**2 + df[' y[kpc]']**2 + df[' z[kpc]']**2)
    return df

def remove_first_timestep(df):
    ####    Description     ####
    # This function removes the first timestep from the dataframe
    # This is because it messes with our analysis
    # Input: df - dataframe of the simulation data
    # Output: df - dataframe of the simulation data with the first timestep removed
    df = df.loc[df['timestep_val'] != 0.0]
    return df

#### ------------------- Plots ------------------- ####

def plot_density(df, timestep, particle_type):
    ####    Description     ####
    # This function plots the density of the particles
    # Input: df - dataframe of the simulation data
    #        timestep - timestep of the simulation
    #        particle_type - type of particle
    # Output: plot of the density of the particles

    #filter the df by timestep and particle type
    df = df.loc[(df['timestep_val'] == timestep) & (df[' particles type'] == particle_type)]
    sns.jointplot(x=' x[kpc]', y=' y[kpc]', data=df, kind='hex')
    #Place the title above the whole plot
    #Scale so x and y limits are 10 10
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    
    plt.title(f'Timestep {timestep}, Particle Type {particle_type}')
    plt.show()


def velocity_dispersion(df, particle_type, timestep):
    ####    Description     ####
    # This function calculates the velocity 
    # dispersion of the particles in the simulation
    # Input: df - dataframe of the simulation data
    # Output: v_dispersion - velocity dispersion of the particles
    #   The velocity dispersion tells us how much the 
    # velocities of the particles vary from the mean velocity

    #Filter the df by particle type and timestep
    df = df.loc[(df[' particles type'] == particle_type) & (df['timestep_val'] == timestep)]

    vx_mean = np.mean(df['vx[km/s]'])
    vy_mean = np.mean(df[' vy[km/s]'])
    vz_mean = np.mean(df[' vz[km/s]'])
    v_dispersion = np.sqrt(np.mean((df['vx[km/s]'] - vx_mean)**2 + (df[' vy[km/s]'] - vy_mean)**2 + (df[' vz[km/s]'] - vz_mean)**2))
    return v_dispersion


def velocity_dispersion_pivot(df):
    ####    Description     ####
    # This function calculates the velocity
    # dispersion of the particles in the simulation
    # this uses the function velocity_dispersion 
    # which is outlined just above
    # and returns a pivot table for the velocity dispersion
    # for each timestep and particle type so it can be compared.

    # Use the velocity_dispersion function to calc velocity dispersion of all timestep and particle type. Place into a pivot table
    #Create a list of all the particle types
    particle_types = df[' particles type'].unique()

    #Create a list of all the timesteps
    timesteps = df['timestep_val'].unique()

    #Create an empty list to store the velocity dispersions
    v_dispersion_list = []

    #Loop through the particle types and timesteps and calculate the velocity dispersion
    for particle_type in particle_types:

        for timestep in timesteps:
            v_dispersion = velocity_dispersion(df, particle_type, timestep)
            v_dispersion_list.append([particle_type, timestep, v_dispersion])

    #Create a dataframe from the list
    v_dispersion_df = pd.DataFrame(v_dispersion_list, columns=['particle_type', 'timestep', 'v_dispersion'])

    #Create a pivot table from the dataframe
    v_dispersion_pivot = v_dispersion_df.pivot(index='timestep', columns='particle_type', values='v_dispersion')

    return v_dispersion_pivot


#Make a gif from the images in plots_3d in order of timestep

def make_gif(file_num, path = '/plots_3d' or '/plots_density',kind= 'quiver' or 'density'):
    file_path = file_path_start+str(file_num)+file_path_end
    #Import the relevant modules
    import imageio
    import os
    #Create a list of the images in plots_3d

    images = []
    frames = len([name for name in os.listdir(file_path+path) if name.endswith('.png')])

    for filename in sorted(os.listdir(file_path+path)):
            if not filename.endswith(".jpg") and not filename.endswith(".png"):
                continue
            # 
            images.append(imageio.imread(file_path+path+'/'+filename))
    #Save the images as a gif in the plots_3d folder as well as a general location in batchrunner
    imageio.mimsave(file_path+path+'/'+kind+'_movie.gif', images, duration=0.5, fps=frames)

    #Create a new variable 'savepath' which is the path to the general location in batchrunner - ie the path stops after /batchrunner/ in file_path
    savepath = file_path[:file_path.find('batchrunner')+12]
    #create a new folder in savepath to store the gifs in if it doesn't already exist
    if not os.path.exists(savepath+'gifs_'+kind):
        os.makedirs(savepath+'gifs_'+kind)
    #Save the images as a gif in the gifs folder
    imageio.mimsave(savepath+'gifs_'+kind+'/'+str(file_num)+'_'+kind+'_movie.gif', images, duration=0.1, fps=frames)

#Create a function to plot the galaxy at any timestep
def plot_galaxy_quiver(timestep, particle_type, df, type = 'map' or 'rb'):

    #Create a new DataFrame for the particles of type 4 at the given timestep
    df_filtered = df[(df['timestep_val'] == timestep) & (df[' particles type'] == particle_type)]

    #Create a quiver plot of the galaxy
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #Make a df which is only the particles of that chosen to have the colours constant over time
    df_sel = df[df[' particles type'] == particle_type]
    #Make a variable 'colours' which is coloured based on the length of the vector arrows.
    norm = plt.Normalize(df_sel['len'].min(), df_sel['len'].max())

    if type == 'rb':
        #Make a variable 'colours' which is coloured based on the direction of the vector arrows. The more positive z is, make it blue. The more negative, make it red. Make these colours slightly translucent
        colours = np.array([(0,0,1,0.5) if z > 0 else (1,0,0,0.5) for z in df_filtered[' y[kpc]']])
    else:
        colours = cm.ScalarMappable(norm=norm, cmap='viridis').to_rgba(df_filtered['len'])

    #Set the axies so it is + and - 10 kpc
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_zlim(-10,10)
    # Plot the quiver plot with the mapped colors
    ax.quiver(df_filtered[' x[kpc]'], df_filtered[' y[kpc]'], df_filtered[' z[kpc]'], 
              df_filtered['vx[km/s]'], df_filtered[' vy[km/s]'], df_filtered[' vz[km/s]'], 
               length =0.1, pivot='middle', color=colours)
    
    #add a colour bar to the plot
    if type == 'rb':
        #Do nothing just continue one
        pass
    else:
        sm = cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        fig.colorbar(sm)
    
    #add axis labels
    ax.set_xlabel('x (kpc)')
    ax.set_ylabel('y (kpc)')
    ax.set_zlabel('z (kpc)')
    
    #Set the title of the plot to the timestep
    plt.title('Timestep = {}'.format(timestep))

def quiver_save(df,particle_type, type, file_num):
    file_path = file_path_start+str(file_num)+file_path_end
    #Create a new folder in file_path to store our plots in if it doesn't already exist
    if not os.path.exists(file_path+'/plots_3d'):
        os.makedirs(file_path+'/plots_3d')
    #loop through each timestep and call plot_galaxy for each timestep and save each to the plots_3d using only the actual timestep valies
    for timestep in df['timestep_val'].unique():
        plot_galaxy_quiver(timestep, particle_type, df, type)
        plt.savefig(file_path+'/plots_3d/{}_galaxy_timestep.png'.format(timestep))
        plt.close()

def calculate_relative_density(df, particle_type):
    # Initialize empty list to store relative densities
    # Note this used to calculate relative densities comparing the first
    # part to later on - but this does not work. So instead 
    # it needs to just be compared to the control later on before
    # it can be assessed for anything 'relative'
    relative_densities = []
    #filter the particle_type
    df = df[df[' particles type'] == particle_type]
    #Find filepath

    #Find the number of timesteps
    timesteps = df['timestep_val'].unique()

    
    # Loop over all other timesteps
    for t in timesteps:
        # Calculate the maximum radius at this timestep
        max_radius = df[df['timestep_val'] == t][[' x[kpc]', ' y[kpc]', ' z[kpc]']].max(axis=1).max()
        
        # Calculate the volume at this timestep
        volume = 4/3 * np.pi * max_radius**3

        # Calculate the density at this timestep
        density = df[df['timestep_val'] == t].shape[0] / volume

        # Calculate the relative density and append to list
        relative_densities.append(density)

    return timesteps, relative_densities

def calculate_relative_density_all_types(df, file_num):
    # Initialize dictionary to store relative densities for each particle type
    relative_densities_dict = {}

    # Define particle type dictionary for legend
    particle_type_dict = {
        0: 'gal1 halo',
        1: 'gal1 bulge',
        2: 'gal1 disk',
        3: 'gal2 halo',
        4: 'gal2 bulge',
        # Continue with other particle types
    }

    # Find filepath
    file_path = file_path_start+str(file_num)+file_path_end

    # Find the number of timesteps
    timesteps = df['timestep_val'].unique()

    # Loop over all particle types
    for particle_type in df[' particles type'].unique():
        # Filter the particle_type
        df_type = df[df[' particles type'] == particle_type]


        # Initialize empty list to store relative densities for this particle type
        relative_densities = []

        # Get the reference timesteps for the first 1 timesteps #Note it is pssible to change this by changing the value of '1' here
        reference_timesteps = timesteps[:1]
        reference_densities = []

        # Calculate the reference density as the average density from timestep 0 to 0.5
        for t in reference_timesteps:
            # Calculate the maximum radius at this timestep
            max_radius = df_type[df_type['timestep_val'] == t][[' x[kpc]', ' y[kpc]', ' z[kpc]']].max(axis=1).max()

            # Calculate the volume at this timestep
            volume = 4/3 * np.pi * max_radius**3

            # Calculate the density at this timestep
            density = df_type[df_type['timestep_val'] == t].shape[0] / volume

            # Append to the reference densities list
            reference_densities.append(density)

        # Calculate the reference density
        reference_density = np.mean(reference_densities)

        # Loop over all timesteps
        for t in timesteps:
            # Calculate the maximum radius at this timestep
            max_radius = df_type[df_type['timestep_val'] == t][[' x[kpc]', ' y[kpc]', ' z[kpc]']].max(axis=1).max()

            # Calculate the volume at this timestep
            volume = 4/3 * np.pi * max_radius**3

            # Calculate the density at this timestep
            density = df_type[df_type['timestep_val'] == t].shape[0] / volume

            # Calculate the relative density and append to list
            relative_density = density / reference_density
            relative_densities.append(relative_density)

        # Add relative densities for this particle type to the dictionary
        relative_densities_dict[particle_type] = relative_densities

    # Plot the relative densities over time for all particle types
    # particle type, relative densities in relative_densities_dict.items():
    for particle_type, relative_densities in relative_densities_dict.items():
        plt.plot(timesteps, relative_densities, label=particle_type_dict.get(particle_type, particle_type))

    #Limit the relative density to 1.5
    plt.ylim(0, 1.5)
    
    plt.xlabel('Timestep')
    plt.ylabel('Relative Density')
    plt.legend()
    plt.title('Relative Density of Galaxy Over Time for File Number {}'.format(file_num))
    #plt.show()

    # Save the plot
    savepath = file_path[:file_path.find('batchrunner')+12]
    kind = 'density_time_all_types'
    # Create a new folder in savepath to store the graphs in if it doesn't already exist
    if not os.path.exists(savepath+'graph_'+kind):
        os.makedirs(savepath+'graph_'+kind)
    # Save the image 
    plt.savefig(savepath+'graph_'+kind+'/'+str(file_num)+'_'+kind+'.png')
    print(savepath+'graph_'+kind+'/'+str(file_num)+'_'+kind+'.png')
    return



#### ------------------- Calculating Paramaters ------------------- ####

import pandas as pd

def calculate_eccentricity(df, particle_type, file_num):
    ####    Description     ####
    # This function calculates the average eccentricity of the particles in a 2D shape (x and y) 
    # across a range of timesteps
    # Input: df - dataframe of the simulation data
    #        particle_type - type of particle
    #        file_num - file number
    # Output: eccentricities - dataframe with eccentricity, particle type, timestep, and file_num
    
    # Filter the dataframe for the particle type
    df = df[df[' particles type'] == particle_type].copy()
    

    # Initialize lists to store the eccentricity, timestep, and file_num
    eccentricities = []
    timesteps = []
    file_nums = []
    
    # Get the unique timesteps in the filtered dataframe
    unique_timesteps = df['timestep_val'].unique()
    
    # For each timestep, calculate the eccentricity and store it in the lists
    for timestep in unique_timesteps:
        df_timestep = df[df['timestep_val'] == timestep]
        
        # Removing outliers using Convex Hull if the number of points is greater than the dimension
        if df_timestep.shape[0] > df_timestep.shape[1]:
            hull = ConvexHull(df_timestep[[' x[kpc]', ' y[kpc]']])
            df_timestep = df_timestep.iloc[hull.vertices]
        
        # Performing PCA
        pca = PCA(n_components=2)
        pca.fit(df_timestep[[' x[kpc]', ' y[kpc]']])
        
        # The explained variance ratios of the PCA components correspond to their lengths
        # We take the ratio of the lengths to calculate the eccentricity
        eccentricity = np.sqrt(1 - pca.explained_variance_ratio_[1] / pca.explained_variance_ratio_[0])
        
        eccentricities.append(eccentricity)
        timesteps.append(timestep)
        file_nums.append(file_num)
    
    # Create a dataframe with eccentricity, particle type, timestep, and file_num
    eccentricity_df = pd.DataFrame({' particles type': particle_type, 'timestep_val': timesteps, file_num+ 'calculate_eccentricity': eccentricities, 'folder': file_nums})
    
    return eccentricity_df


def gaussian(x, a, x0, sigma):
    ####    Description     ####
    # This function creates a gaussian curve
    # Input: x - x values
    #        a - amplitude
    #        x0 - mean
    #        sigma - standard deviation
    # Output: gaussian curve
    #Note this is to be used in calculating the fwhm both 2d and 3d

    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def calculate_fwhm_3D(df, particle_type, folder):
    ####    Description     ####
    # This function calculates the Full Width at Half Maximum 
    # (FWHM) of the radial distribution
    # Input: df - dataframe of the simulation data
    #        particle_type - type of particle
    #        timestep - timestep of the simulation
    # Output: fwhm - Full Width at Half Maximum (FWHM)
    #  of the radial distribution

    #Get the range of timesteps
    timesteps = df['timestep_val'].unique()
    # Initialize lists to store the FWHM, timestep, and file_num
    fwhms = []
    timesteps_list = []
    file_nums = []
    
    # Loop over the timesteps and calculate the FWHM for each
    for timestep in timesteps:
        # Create a new DataFrame for the particles of the given particle_type and timestep
        df_timestep = df[(df['timestep_val'] == timestep) & (df[' particles type'] == particle_type)].copy()

        """Calculate Full Width at Half Maximum (FWHM) of radial distribution"""
        # Calculate distances from the center
        distances = np.sqrt(df_timestep[' x[kpc]']**2 + df_timestep[' y[kpc]']**2 + df_timestep[' z[kpc]']**2)
        
        # Compute a histogram of these distances
        histogram, bin_edges = np.histogram(distances, bins='auto', density=False)
        
        # Fit a Gaussian to the histogram
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        popt, _ = curve_fit(gaussian, bin_centers, histogram, p0=[histogram.max(), distances.mean(), distances.std()])
        
        # FWHM is 2*sqrt(2*log(2))*sigma
        fwhm = 2*np.sqrt(2*np.log(2))*popt[2]
        
        # Store the FWHM, timestep, and file_num in the lists
        fwhms.append(fwhm)
        timesteps_list.append(timestep)
        file_nums.append(folder)
    
    # Create a dataframe with FWHM, particle type, timestep, and file_num
    fwhm_df = pd.DataFrame({
        ' particles type': particle_type,
        'timestep_val': timesteps_list,
        folder + 'calculate_fwhm_3D': fwhms,
        'folder': file_nums
    })
    
    return fwhm_df

def calculate_fwhm_2D(df, particle_type, folder):
    ####    Description     ####
    # This function calculates the Full Width at Half Maximum
    # (FWHM) of the radial distribution in the x-y plane
    # Input: df - dataframe of the simulation data
    #        particle_type - type of particle
    #        timestep - timestep of the simulation
    # Output: fwhm - Full Width at Half Maximum (FWHM)
    #  of the radial distribution in the x-y plane

    #Get the range of timesteps
    timesteps = df['timestep_val'].unique()
    # Initialize lists to store the FWHM, timestep, and file_num
    fwhms = []
    timesteps_list = []
    file_nums = []
    
    # Loop over the timesteps and calculate the FWHM for each
    for timestep in timesteps:
        # Create a new DataFrame for the particles of the given particle_type and timestep
        df_timestep = df[(df['timestep_val'] == timestep) & (df[' particles type'] == particle_type)].copy()


        """Calculate Full Width at Half Maximum (FWHM) of radial distribution in x-y plane"""
        # Calculate distances from the center in x-y plane
        distances = np.sqrt(df_timestep[' x[kpc]']**2 + df_timestep[' y[kpc]']**2)

        # Compute a histogram of these distances
        histogram, bin_edges = np.histogram(distances, bins='auto', density=False)

        # Fit a Gaussian to the histogram
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        popt, _ = curve_fit(gaussian, bin_centers, histogram, p0=[histogram.max(), distances.mean(), distances.std()])

        # FWHM is 2*sqrt(2*log(2))*sigma
        fwhm = 2*np.sqrt(2*np.log(2))*popt[2]

        # Store the FWHM, timestep, and file_num in the lists
        fwhms.append(fwhm)
        timesteps_list.append(timestep)
        file_nums.append(folder)

    # Create a dataframe with FWHM, particle type, timestep, and file_num
    fwhm_df = pd.DataFrame({
        ' particles type': particle_type,
        'timestep_val': timesteps_list,
        folder + 'calculate_fwhm_2D': fwhms,
        'folder': file_nums
    })

    return fwhm_df


def calculate_fwhm_method_1(df, particle_type, folder):
    ####    Description     ####
    # This function calculates the Full Width at Half Maximum
    # (FWHM) by finding it on the histogram
    # Input: df - dataframe of the simulation data
    #        particle_type - type of particle
    #        timestep - timestep of the simulation
    # Output: fwhm - Full Width at Half Maximum (FWHM)
    #  of the radial distribution in the x-y plane
    # Note this is just to compare the different methods for now
    #Get the range of timesteps

    timesteps = df['timestep_val'].unique()
    # Initialize lists to store the FWHM, timestep, and file_num
    fwhms = []
    timesteps_list = []
    file_nums = []
# Loop over the timesteps and calculate the FWHM for each
    for timestep in timesteps:
        # Create a new DataFrame for the particles of the given particle_type and timestep
        df_timestep = df[(df['timestep_val'] == timestep) & (df[' particles type'] == particle_type)].copy()

        df_timestep = np.sqrt(df_timestep[' x[kpc]']**2 + df_timestep[' y[kpc]']**2 + df_timestep[' z[kpc]']**2)

        """Calculate Full Width at Half Maximum (FWHM)"""
        histogram, bin_edges = np.histogram(df_timestep, bins='auto', density=True)
        max_value = histogram.max()
        half_max_value = max_value / 2

        # Find where histogram crosses half max
        above_half_max = np.where(histogram > half_max_value)[0]
        if above_half_max.size == 0:
            fwhm = None
        else:
            bin_min = bin_edges[above_half_max[0]]
            bin_max = bin_edges[above_half_max[-1] + 1]
            fwhm = bin_max - bin_min

        # Store the FWHM, timestep, and file_num in the lists
        fwhms.append(fwhm)
        timesteps_list.append(timestep)
        file_nums.append(folder)

    # Create a dataframe with FWHM, particle type, timestep, and file_num
    fwhm_df = pd.DataFrame({
        ' particles type': particle_type,
        'timestep_val': timesteps_list,
        folder + 'calculate_fwhm_method_1': fwhms,
        'folder': file_nums
    })

    return fwhm_df



def calculate_relative_density_table(df, particle_type, folder):
    ####    Description     ####
    # This function calculates the average relative density
    # of the galaxy over a given range of timesteps
    # Input: df - dataframe of the simulation data
    #        particle_type - type of particle
    #        folder - folder name
    # Output: relative_density_df - dataframe with relative density, particle type, and folder
    
    timesteps, relative_densities = calculate_relative_density(df, particle_type)
    relative_density_df = pd.DataFrame({
        ' particles type': particle_type,
        'timestep_val':timesteps,
        'folder': folder,
        folder+ 'calculate_relative_density_table': relative_densities
    })
    
    return relative_density_df



def calculate_metrics(df, particle_types, folder):
    timesteps = df['timestep_val'].unique()

    # Initialize an empty DataFrame to store the results
    df_metrics = pd.DataFrame(columns=[' particles type', 'timestep_val', 'folder', folder+'position_std_dev', folder+'mean_velocity', folder+'velocity_std_dev'])

    for particle_type in particle_types:
        for timestep in timesteps:
            #skip timestep 0.0 if it exists
            if timestep == 0.0:
                continue
            # Filter the DataFrame for the current particle type and timestep
            df_filtered = df[(df[' particles type'] == particle_type) & (df['timestep_val'] == timestep)]
            
            # Calculate the standard deviation of position
            position_std_dev = df_filtered[[' x[kpc]', ' y[kpc]', ' z[kpc]']].std().mean()

            # Calculate the mean and standard deviation of velocity
            mean_velocity = np.sqrt((df_filtered['vx[km/s]']**2 + df_filtered[' vy[km/s]']**2 + df_filtered[' vz[km/s]']**2).mean())
            velocity_std_dev = np.sqrt((df_filtered['vx[km/s]']**2 + df_filtered[' vy[km/s]']**2 + df_filtered[' vz[km/s]']**2).std())

            # Append the results to the DataFrame
            df_metrics = df_metrics.append({
                ' particles type': particle_type,
                'timestep_val': timestep,
                'folder': folder,
                folder+'position_std_dev': position_std_dev,
                folder+'mean_velocity': mean_velocity,
                folder+'velocity_std_dev': velocity_std_dev
            }, ignore_index=True)
    return df_metrics



def calculate_surface_brightness(df, particle_type, folder):
    results = []
    # Get the unique timesteps for the particle type
    timesteps = df[df[' particles type'] == particle_type]['timestep_val'].unique()

    for timestep in timesteps:
        df_timestep = df[(df[' particles type'] == particle_type) & (df['timestep_val'] == timestep)].copy()

        x_bins = np.linspace(df_timestep[' x[kpc]'].min(), df_timestep[' x[kpc]'].max(), 50)
        y_bins = np.linspace(df_timestep[' y[kpc]'].min(), df_timestep[' y[kpc]'].max(), 50)

        histogram, x_edges, y_edges = np.histogram2d(df_timestep[' x[kpc]'], df_timestep[' y[kpc]'], bins=(x_bins, y_bins))
        surface_brightness = histogram / (x_edges[1] - x_edges[0]) / (y_edges[1] - y_edges[0])

        average_surface_brightness = surface_brightness.mean()

        results.append([particle_type, timestep, folder, average_surface_brightness])

    results_df = pd.DataFrame(results, columns=[' particles type', 'timestep_val', 'folder', folder + '_calculate_surface_brightness'])
    return results_df


def calculate_half_mass_radius(df, particle_type, folder):
    # Get the unique timesteps for the particle type
    timesteps = df[df[' particles type'] == particle_type]['timestep_val'].unique()
    
    df = df[df[' particles type'] == particle_type]
    df['r'] = np.sqrt(df[' x[kpc]']**2 + df[' y[kpc]']**2 + df[' z[kpc]']**2)
    df = df.sort_values('r')
    
    results = []
    for timestep in timesteps:
        df_timestep = df[df['timestep_val'] == timestep]
        cumulative_mass = np.cumsum(np.ones(len(df_timestep)))
        half_mass_index = np.argmin(np.abs(cumulative_mass - 0.5 * len(df_timestep)))
        half_mass_radius = df_timestep.iloc[half_mass_index]['r']
        results.append([particle_type, timestep, folder, half_mass_radius])
    
    results_df = pd.DataFrame(results, columns=[' particles type', 'timestep_val', 'folder', folder + '_calculate_half_mass_radius'])
    return results_df


def calculate_scale_length(df, particle_type, folder):
    # Get the unique timesteps for the particle type
    timesteps = df[df[' particles type'] == particle_type]['timestep_val'].unique()
    
    scale_lengths = []
    for timestep in timesteps:
        df_timestep = df[(df[' particles type'] == particle_type) & (df['timestep_val'] == timestep)]
        df_timestep['r'] = np.sqrt(df_timestep[' x[kpc]']**2 + df_timestep[' y[kpc]']**2)
        bins = np.linspace(0, df_timestep['r'].max(), 100)
        histogram, bin_edges = np.histogram(df_timestep['r'], bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        def exponential_disk(r, I_0, h):
            return I_0 * np.exp(-r / h)

        def alternative_disk(r, I_0, h):
            return I_0 / (1 + (r / h))

        try:
            popt, _ = curve_fit(exponential_disk, bin_centers, histogram)
            scale_length = popt[1]
        except RuntimeError:
            popt = least_squares(lambda params: exponential_disk(bin_centers, *params) - histogram, [1, 1]).x
            scale_length = popt[1]

            if np.isnan(scale_length):
                popt = least_squares(lambda params: alternative_disk(bin_centers, *params) - histogram, [1, 1]).x
                scale_length = popt[1]

        scale_lengths.append(scale_length)
    
    results_df = pd.DataFrame({' particles type': particle_type, 'timestep_val': timesteps, 'folder': folder, folder + '_calculate_scale_length': scale_lengths})
    return results_df




start_timestep_1 = 0.01
end_timestep_1 = 1
start_timestep_2 = 4
end_timestep_2 = 6
#results_df = compare_properties(df, start_timestep_1, end_timestep_1, start_timestep_2, end_timestep_2)

# -------- General analysing functions -------#

def analyse_function_range(func, df, particle_types, start_timestep, end_timestep, folder):
    control_raw = pd.DataFrame()  # Initialize an empty dataframe for control_raw
    control_summary = pd.DataFrame()  # Initialize an empty dataframe for control_summary
    
    for particle_type in particle_types:
        try:
            # Get the raw dataframe
            control_raw_particle = func(df, particle_type, folder)
            # Append the raw dataframe to control_raw
            control_raw = control_raw.append(control_raw_particle, ignore_index=True)
            
            # Create a summary dataframe for timesteps between start_timestep and end_timestep
            control_summary_particle = control_raw_particle[(control_raw_particle['timestep_val'] >= start_timestep) & (control_raw_particle['timestep_val'] <= end_timestep)].groupby([' particles type', 'folder']).mean().reset_index()
            
            # Append the summary dataframe to control_summary
            control_summary = control_summary.append(control_summary_particle, ignore_index=True)
        
        except Exception as e:
            print(f"Error occurred for particle type: {particle_type}, folder: {folder}, function: {func.__name__}")
            print(f"Error message: {str(e)}")
            continue  # Continue to the next particle_type if an error occurs

    return control_summary, control_raw


def plot_heatmap(df, particle_type, values_col, kind, type):
    # Filter for specified particle type
    df_filtered = df[df[' particles type'] == particle_type]

    #Cap the outliers in the values_col
    df_filtered = cap_outliers(df_filtered, values_col)

    # Create a pivot table with vx and vy as the x and y axes and specified column as the values
    pivot = df_filtered.pivot_table(index='vx', columns='vy', values=values_col, aggfunc='mean')

    # Create heatmap
    sns.heatmap(pivot, cmap='viridis')
    
    # Add title and axis labels
    plt.title(f'{values_col.capitalize()} of particles type {particle_type}, for simulation {type}')
    plt.xlabel('vx')
    plt.ylabel('vy')

    # Invert y axis
    plt.gca().invert_yaxis()

    # Show plot
    #plt.show()

    #Save the plot in folder 'analysis_summary' and create it if it doesnt exist
    save_folder = os.path.join(current_dir, 'analysis_summary')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(os.path.join(save_folder, f'{values_col.capitalize()} of particles type {particle_type}, for simulation {type}.png'))
    plt.close()


def plot_all_heatmaps(df, particle_types, kind, type):
    # Find all columns ending with '_norm'
    norm_cols = [col for col in df.columns if col.endswith(kind)]

    # Loop over all particle types and norm columns
    for particle_type in particle_types:
        for norm_col in norm_cols:
            plot_heatmap(df, particle_type, norm_col, kind,type)

def cap_outliers(df, column_name, upper_percentile=95, lower_percentile=5):
    upper = np.percentile(df[column_name].values, upper_percentile)
    lower = np.percentile(df[column_name].values, lower_percentile)
    df[column_name] = np.clip(df[column_name], lower, upper)
    return df



def plot_controls(df, values_col, highlight_part):
    # Separate the data for particle type 4 and the rest
    df_type_4 = df[df[' particles type'] == highlight_part]
    df_rest = df

    # Plot the lines for the rest of the particle types
    sns.lineplot(data=df_rest, x='timestep_val', y=values_col, hue=' particles type')

    #select the 5th colour in the colour palette for the cubehelix_palette
    sel_col = sns.cubehelix_palette(5)[highlight_part]
    
    # Plot a thicker, partly transparent line for particle type 4
    plt.plot(df_type_4['timestep_val'], df_type_4[values_col], 
             linewidth=5, alpha=0.3, color=sel_col)  # Adjust these parameters as needed

    # Then plot the normal line for particle type 4
    sns.lineplot(data=df_type_4, x='timestep_val', y=values_col, 
                 color=sel_col, label='4: Gal2 bulge')

    # Customize the title and legend - reduce the size 
    plt.title(f'Plot of {values_col.capitalize()} over time for control in type 1 simulation', fontsize=10)
    plt.legend(title='Particle type', loc='upper right', labels=['0: Gal1 halo', '1: Gal1 bulge', '2: Gal1 disk', '3: Gal2 halo', '4: Gal2 bulge'])
  
    #Add a data call out of the average of the highlighted particle type over timestep start_timestep to end_timestep. Add it to the end of the line. Round the value to 3sf
    avg_value = df_type_4[(df_type_4['timestep_val'] >= start_timestep) & (df_type_4['timestep_val'] <= end_timestep)][values_col].mean()
    plt.text(end_timestep + 0.1, avg_value, round(avg_value, 2), fontsize=10, color=sel_col)

    #Add a line to help compare this value to the rest of the plot
    plt.axhline(y=avg_value, color=sel_col, linestyle='--', label='Mean of highlighted particle type')
    #Show the plot
    #plt.show()
    #Save the plots to a folder 'plots_control_type 1' - if no folder exists, make it
    if not os.path.exists('plots_control_type 1'):
        os.makedirs('plots_control_type 1')
    plt.savefig(f'plots_control_type 1/{values_col}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_controls(df, highlight_part):
    # Find all columns except ' particles type' and 'timestep_val'
    cols = [col for col in df.columns if col not in [' particles type', 'timestep_val']]

    # Loop over columns
    for col in cols:
        plot_controls(df, col, highlight_part)



def plot_heatmap_decide(df, function_name, kind, particle_type, bins, type, order='correct'):
    # Select rows with the given particle type
    df = df[df[' particles type'] == particle_type]
    #drop na rows only when the na is in the selected column
    df = df.dropna(subset=[f'{function_name}{kind}'])
    # Create a pivot table with vx and vy as the axes and the function as the values
    pivot_table = df.pivot_table(index='vx', columns='vy', values=f'{function_name}{kind}')

    
    # Create a colormap for the bins
    #colors = ['green', 'orange', 'red']
    #cmap = mcolors.LinearSegmentedColormap.from_list('Custom', colors, len(colors))
    cmap = ListedColormap(['#b5e876', '#ffbe61', '#ff6161'])
    norm = mcolors.BoundaryNorm(bins, 3)

    if order == 'correct':  # if bins are in ascending order
        colors = ['#b5e876', '#ffbe61', '#ff6161']  # green to red
    elif order == 'opposite':  # if bins are in descending order
        colors = ['#ff6161', '#ffbe61', '#b5e876']  # red to green
    else:
        colors = ['#ff6161', '#b5e876', '#ffbe61']# red green orange
    cmap = ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bins, 3)

    # Plot the heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(pivot_table, cmap=cmap, norm=norm)

    # Add a label for the colours used
    #color_labels = ['Ok', 'Borderline', 'Too distorted']
    colors = [cmap(x) for x in range(cmap.N)]
    #custom_labels = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(color_labels))]

    # Add the color labels to the plot
    #plt.legend(custom_labels, color_labels, title='Status', loc='upper left')

    plt.title(f'Heatmap of {function_name.capitalize()} for Particle Type {particle_type} for {type}')
    
    plt.gca().invert_yaxis()
    #Add axis labels
    plt.xlabel('vx')
    plt.ylabel('vy')
    
    #plt.show()
    #Save the plot in 'analysis' folder - create the folder if it doesn't exist
    if not os.path.exists('analysis'):
        os.makedirs('analysis')
    plt.savefig(f'analysis/{function_name}{kind}_type{particle_type}_{type}.png')
    plt.close()



def plot_controls(df, values_col, highlight_part):
    # Separate the data for particle type 4 and the rest
    df_type_4 = df[df[' particles type'] == highlight_part]
    df_rest = df

    # Plot the lines for the rest of the particle types
    sns.lineplot(data=df_rest, x='timestep_val', y=values_col, hue=' particles type')

    #select the 5th colour in the colour palette for the cubehelix_palette
    sel_col = sns.cubehelix_palette(5)[highlight_part]
    
    # Plot a thicker, partly transparent line for particle type 4
    plt.plot(df_type_4['timestep_val'], df_type_4[values_col], 
             linewidth=5, alpha=0.3, color=sel_col)  # Adjust these parameters as needed

    # Then plot the normal line for particle type 4
    sns.lineplot(data=df_type_4, x='timestep_val', y=values_col, 
                 color=sel_col, label='4: Gal2 bulge')

    # Customize the title and legend - reduce the size 
    plt.title(f'Plot of {values_col.capitalize()} over time for control in type 1 simulation', fontsize=10)
    plt.legend(title='Particle type', loc='upper right', labels=['0: Gal1 halo', '1: Gal1 bulge', '2: Gal1 disk', '3: Gal2 halo', '4: Gal2 bulge'])
  
    #Add a data call out of the average of the highlighted particle type over timestep start_timestep to end_timestep. Add it to the end of the line. Round the value to 3sf
    avg_value = df_type_4[(df_type_4['timestep_val'] >= start_timestep) & (df_type_4['timestep_val'] <= end_timestep)][values_col].mean()
    plt.text(end_timestep + 0.1, avg_value, round(avg_value, 2), fontsize=10, color=sel_col)

    #Add a line to help compare this value to the rest of the plot
    plt.axhline(y=avg_value, color=sel_col, linestyle='--', label='Mean of highlighted particle type')
    #Show the plot
    #plt.show()
    #Save the plots to a folder 'plots_control_type 1' - if no folder exists, make it
    if not os.path.exists('plots_control_type 1'):
        os.makedirs('plots_control_type 1')
    plt.savefig(f'plots_control_type 1/{values_col}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_controls(df, highlight_part):
    # Find all columns except ' particles type' and 'timestep_val'
    cols = [col for col in df.columns if col not in [' particles type', 'timestep_val']]

    # Loop over columns
    for col in cols:
        plot_controls(df, col, highlight_part)



def plot_bases(df, values_col, highlight_part):
    # Separate the data for particle type 4 and the rest
    df_type_sel = df[df[' particles type'] == highlight_part]
    df_rest = df

    # Plot the lines for the rest of the particle types
    sns.lineplot(data=df_rest, x='timestep_val', y=values_col, hue=' particles type')

    #select the 5th colour in the colour palette for the cubehelix_palette
    sel_col = sns.cubehelix_palette(6)[highlight_part]
    
    # Plot a thicker, partly transparent line for the selected particle
    plt.plot(df_type_sel['timestep_val'], df_type_sel[values_col], 
             linewidth=5, alpha=0.3, color=sel_col)  # Adjust these parameters as needed

    # Then plot the normal line for the selected particle
    sns.lineplot(data=df_type_sel, x='timestep_val', y=values_col, 
                 color=sel_col)

    # Customize the title and legend - reduce the size 
    plt.title(f'Plot of {values_col.capitalize()} over time for base in type 2 simulation', fontsize=10)
    plt.legend(title='Particle type', loc='upper right', labels=['0: Gal1 halo', '1: Gal1 bulge', '2: Gal1 disk', '3: Gal2 halo', '4: Gal2 bulge', '5: Gal2 disk'])
  
    #Add a data call out of the average of the highlighted particle type over timestep start_timestep to end_timestep. Add it to the end of the line. Round the value to 3sf
    avg_value = df_type_sel[(df_type_sel['timestep_val'] >= start_timestep) & (df_type_sel['timestep_val'] <= end_timestep)][values_col].mean()
    plt.text(end_timestep + 0.1, avg_value, round(avg_value, 2), fontsize=10, color=sel_col)
    #If the values_col is scale length, limit the y axis to 60
    if values_col == 'calculate_scale_length': # Note I added this manually in as the data was just like that here
        plt.ylim(0, 60)
        
    #Add a line to help compare this value to the rest of the plot
    plt.axhline(y=avg_value, color=sel_col, linestyle='--', label='Mean of highlighted particle type')
    #Show the plot
    #plt.show()
    #Save the plots to a folder 'plots_base_type 1' - if no folder exists, make it
    if not os.path.exists('plots_base_type_2'):
        os.makedirs('plots_base_type_2')
    plt.savefig(f'plots_base_type_2/{values_col}_{highlight_part}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_bases(df, highlight_part):
    # Find all columns except ' particles type' and 'timestep_val'
    cols = [col for col in df.columns if col not in [' particles type', 'timestep_val']]

    # Loop over columns
    for col in cols:
        plot_bases(df, col, highlight_part)




#### ------------------ Summarising the data -------------- ####

def call_headers_summary():
    #This finds the columns ranges and orders we have chosen from the data
    data = {
        'Plot Name': [
            'calculate_relative_density_table_norm',
            'calculate_scale_length_norm',
            'calculate_fwhm_method_1_norm',
            'calculate_fwhm_3D_norm',
            'calculate_fwhm_2D_norm',
            'calculate_half_mass_radius_norm',
            'calculate_eccentricity_data'
        ],
        'Range': [
            '[0, 0.001, 0.01, 1]',
            '[0, 1.2, 2, 10]',
            '[0, 1.5, 5, 10]',
            '[0, 1.5, 5, 50]',
            '[0, 1.5, 5, 50]',
            '[0, 1.5, 5, 10]',
            '[0, 0.9, 0.95, 1]'
        ],
        'Order': [
            'Correct',
            'Middle',
            'Middle',
            'Middle',
            'Middle',
            'Middle',
            'Correct'
        ]
    }

    df_headers = pd.DataFrame(data)
    return df_headers

def assign_summary(row, bins, order):
    if order == 'correct':
        if bins[0] <= row < bins[1]:
            return 'y'
        elif bins[1] <= row < bins[2]:
            return 'm'
        elif bins[2] <= row <= bins[3]:
            return 'n'
    else:  # assuming 'middle'
        if bins[0] <= row < bins[1]:
            return 'n'
        elif bins[1] <= row < bins[2]:
            return 'y'
        elif bins[2] <= row <= bins[3]:
            return 'm'
    return np.nan  # return NaN for values outside the bins

def generate_summary(norm_data_init_cl, df_headers):
    df_headers['Range'] = df_headers['Range'].apply(ast.literal_eval)  # convert range strings to lists
    
    # Create a new dataframe with the required columns
    summary_df = norm_data_init_cl[[' particles type', 'vx', 'vy']].copy()

    for _, row in df_headers.iterrows():
        plot_name = row['Plot Name']
        bins = row['Range']
        order = row['Order'].lower()

        summary_col_name = plot_name.replace('_norm', '').replace('_data', '') + '_summary'
        summary_df[summary_col_name] = norm_data_init_cl[plot_name].apply(assign_summary, args=(bins, order))
    
    return summary_df



def summarise_data(row, columns_to_check):
    """
    The function checks if all the required columns in a given row 
    have 'y' or 'm'. If all are 'y', it returns 'y', if all are 'm', 
    it returns 'm', else it returns 'n'.
    """
    if all(row[col] == 'y' for col in columns_to_check):
        return 'y'
    elif all(row[col] == 'm' for col in columns_to_check):
        return 'm'
    else:
        return 'n'

def summarise_data_broad(row, columns_to_check):
    """
    Based on the condition of the column 'calculate_eccentricity_summary',
    and the other specified columns in the row, this function returns 
    'y', 'm' or 'n'.
    """
    for col in columns_to_check:
        if row['calculate_eccentricity_summary'] == 'y' and row[col] == 'y':
            return 'y'
        elif row['calculate_eccentricity_summary'] == 'm' and row[col] == 'm':
            return 'm'
        elif row['calculate_eccentricity_summary'] in ('y', 'm') and row[col] in ('y', 'm'):
            return 'm'
    return 'n'

def summarise_data_y(row, columns_to_check):
    """
    This function counts the number of 'y' in each row of the specified 
    columns except for 'calculate_eccentricity_summary'.
    """
    return sum(1 for col in columns_to_check if row[col] == 'y')

def summarise_data_m(row, columns_to_check):
    """
    This function counts the number of 'm' in each row of the specified 
    columns except for 'calculate_eccentricity_summary'.
    """
    return sum(1 for col in columns_to_check if row[col] == 'm')

def summarise_data_narrow(row, columns_to_check):
    """
    This function returns 'y' if 'calculate_eccentricity_summary' is 'y' 
    and the count in 'pre_summary_y' is 2 or more, 'm' if 
    'calculate_eccentricity_summary' is 'm' and 'pre_summary_y' or 
    'pre_summary_m' is 2 or more, else it returns 'n'.
    """
    if row['calculate_eccentricity_summary'] == 'y' and row['summary_y'] >= 2:
        return 'y'
    elif row['calculate_eccentricity_summary'] == 'm' and (row['summary_y'] >= 2 or row['summary_m'] >= 2):
        return 'm'
    else:
        return 'n'

def replace_values(df, column_name):
    new_column_name = column_name + '_num'
    df[new_column_name] = df[column_name].replace('y', 0.5)
    df[new_column_name] = df[new_column_name].replace('m', 1.5)
    df[new_column_name] = df[new_column_name].replace('n', 2.5)
    return df


# Data-Mining-Project
Transformation of Data Code. To add compression to the GTZAN wav files we must run the compress_data.m file. 
For this code we use the Audio Toolbox in Matlab. This contains many audio engineering tools like compression and equalization. 
We loop through all of the files in the data folder and apply the dynamic range compressor to each file. We give the 
file a new name that is the old name but with compressed in front of it. Then we write the output file to a folder called 
compressed data. The ratio is changed from 10:1 to 2:1 for the second batch of compressed data. We use the same process for 
the equalization effect.

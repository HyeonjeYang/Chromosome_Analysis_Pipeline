# Chromosome_Analysis_Pipeline
Pipelines for analyzing chromosome organization : TADs, compartments... etc.
<br>
Work in Progress... (last modified : 02/25/2026)
<br>
This project is based on the open-source repository:<br>
https://github.com/open2c/polychrom.git
<br>
Licensed under the MIT License.
<br>
<br>
Hi-C modeling and Chromosome Visualization.
<br>
How to use? --> Get ".h5" file from polychrom simulation.
<br>
You can download the example file created using polychrom (blocks_0-99.h5) and run the simulation!
<br>
It takes quite long. So keep in mind to set the xlim and ylim properly.
<br>
<br>
You can also visualize the chromosome organization using blender.
<br>
.csv (coordinates.h5 --> coordinates.csv)
<br>
.npy also recommended
<br>
see the "visulization" directory.
<br>
<br>
For real and more precise work pipeline, please look into the "work_pipeline" directory.
<br>
(You can make Hi-C Map, P(s) plot, and gamma plot!)
<br>
<br>
If you are using GPU, I recommend that you run Loop_Extrusion_3D_simu.py and polychrom_simu_get_h5.py in your terminal.
<br>
python filename.py --> you can run!
<br>
<br>
Thank you! Please Enjoy!

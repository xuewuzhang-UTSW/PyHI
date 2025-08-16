# PyHI
V007: It now can read a 3D map from a helical reconstruction, and generate 2D projection images and power spectra. It is useful for learning. 
Also fixed a bug that leads to incorrect calculation of the displayed repeat distance in Tab1 (which did affect the indexing or the final outputs of rise and twist).

New version of the average power spectrum calculator, allowing specifiying the rotation angle of particles to correct imperfect alignmment of 2D class average to the horizontal axis from RELION. Also added multiprocessing to make it run faster, and it now also write out a realspace image of the 2D average.

V006: Tab1: can read stacks (.mrc or .mrcs) of power spectra; Tab2: can highlight Bessel peaks and their Bessel orders based on the current refined lattice. 
(Note on 1/28/2025: Thanks to Ruizhi Peng, a bug (issue #5) was noticed and fixed).

Note: V005 addes a new tab which generates a 3D representation of the helix and Relion command for making initial model. This needs a newer version of Matplotlib (>3.6).


Written by Xuewu Zhang, Department of Pharmacology, University of Texas Southwestern Medical Center, Dallas, USA.

A video demo on how to use it:
https://www.youtube.com/watch?v=KxAeo90CIt4

A more detailed description can be found at (Citation:Xuewu Zhang, Protein Sci. 2021. doi: 10.1002/pro.4186):
https://onlinelibrary.wiley.com/doi/10.1002/pro.4186

This is a Python-based GUI program for indexing power spectra of helical assemblies. The program is a Python script that runs under Python version 3.7. The script depends on the following Python libraries: Mrcfile 1.1.2 (Burnley et al., 2017), numpy 1.18.3 (Harris et al., 2020), matplotlib 3.4.2 (Hunter, 2007), Pillow 7.2.0 (https://pillow.readthedocs.io/en/stable/), PyQt5 5.15.0 (https://pypi.org/project/PyQt5/), SciPy 1.4.1 (Virtanen et al., 2020) and mplsursors 0.4 (https://mplcursors.readthedocs.io/en/stable/). Newer versions of these packages should work as well. These libraries can be installed on modern operating systems with standard Python library management tools such as PIP (Python Package Installer; https://pip.pypa.io/en/stable/) and anaconda (https://docs.conda.io/en/latest/). 

For example, you can download Miniconda from here:
https://docs.conda.io/en/latest/miniconda.html

Follow the intruction to install the python 3 version of Miniconda. To activate Conda:

$ source /directory/to/miniconda/bin/activate 


Now you can directly install the dependency packages by using commands such as:

$ conda install numpy

Or you can create a virtual environment for PyHI:

$ conda create --name PyHI


To activate this new virtual environment:

$ source /directory/to/miniconda/bin/activate PyHI


Now you can install the packages within this envirnoment. To run PyHI, change to the directory where PyHI_v003.py resides and then:

$ python3 PyHI_v004.py



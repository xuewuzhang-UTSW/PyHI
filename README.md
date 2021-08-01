# PyHI
Written by Xuewu Zhang, Department of Pharmacology, University of Texas Southwestern Medical Center, Dallas, USA.

This is a Python-based GUI program for indexing power spectra of helical assemblies. The program is a Python script that runs under Python version 3.7. The script depends on the following Python libraries: Mrcfile 1.1.2 (Burnley et al., 2017), numpy 1.18.3 (Harris et al., 2020), matplotlib 3.3.1 (Hunter, 2007), Pillow 7.2.0 (https://pillow.readthedocs.io/en/stable/), PyQt5 5.15.0 (https://pypi.org/project/PyQt5/), SciPy 1.4.1 (Virtanen et al., 2020) and mplsursors 0.4 (https://mplcursors.readthedocs.io/en/stable/). These libraries can be installed on modern operating systems with standard Python library management tools such as PIP (Python Package Installer; https://pip.pypa.io/en/stable/) and anaconda (https://docs.conda.io/en/latest/). 

For example, you can download Miniconda from here:
https://docs.conda.io/en/latest/miniconda.html

Follow the intruction to install the python 3 version of Miniconda. To activate Conda:

$ source /directory/to/miniconda/bin/activate 


Now you can directly install the dependency packages by using commands such as:

conda install numpy

Or you can create a virtual environment for PyHI:

conda create --name PyHI


To activate this new virtual environment:

source /directory/to/miniconda/bin/activate PyHI


Now you can install the packages within this envirnoment. To run PyHI, change to the directory where PyHI_v003.py resides and then:

python3 PyHI_v003.py



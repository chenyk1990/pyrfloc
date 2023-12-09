**pyrfloc**
======

## Description

**pyrfloc** is a python package of RFloc3D method for passive seismic location using machine learning (Random Forest).



## Reference
    Chen, Y., Saad, O. M., Savvaidis, A., Chen, Y., & Fomel, S. (2022). 3D Microseismic Monitoring Using Machine Learning. Journal of Geophysical Research: Solid Earth, 127(3), e2021JB023842.

    Chen, Y., A. Savvaidis, Fomel, S., Saad, O.M. and Y.F. Chen, 2023. RFloc3D: A Machine- Learning Method for 3-D Microseismic Source Location Using P-and S-Wave Arrivals: IEEE Transactions on Geoscience and Remote Sensing, 61, 5901310.
    
BibTeX:

	@article{rfloc3d,
	  title={3{D} microseismic monitoring using machine learning},
	  author={Yangkang Chen and Omar M. Saad and Alexandros Savvaidis and Yunfeng Chen and Sergey Fomel},
	  journal={Journal of Geophysical Research - Solid Earth},
	  volume={127},
	  number={3},
	  issue={3},
	  pages={e2021JB023842},
	  year={2022}
	}

	@Article{rfloc3d,
	  author={Yangkang Chen and Alexandros Savvaidis and Sergey Fomel and Omar M. Saad and Yunfeng Chen},
	  title = {RFloc3{D}: a machine learning method for 3{D} microseismic source location using P- and S-wave arrivals},
	  journal={IEEE Transactions on Geoscience and Remote Sensing},
	  year=2023,
	  volume=61,
	  issue=1,
	  number=1,
	  pages={5901310},
	  doi={10.1109/TGRS.2023.3236572},
	}

-----------
## Copyright
	The pyrfloc developing team, 2021-present
-----------

## License
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)   

-----------

## Install
Using the latest version

    git clone https://github.com/chenyk1990/pyrfloc
    cd pyrfloc
    pip install -v -e .
or using Pypi

    pip install pyrfloc

-----------
## Examples
    The "demo" directory contains all runable scripts to demonstrate different applications of pyrfloc. 

-----------
## Dependence Packages
* scipy 
* numpy 
* matplotlib

-----------
## Development
    The development team welcomes voluntary contributions from any open-source enthusiast. 
    If you want to make contribution to this project, feel free to contact the development team. 

-----------
## Contact
    Regarding any questions, bugs, developments, collaborations, please contact  
    Yangkang Chen
    chenyk2016@gmail.com

-----------
## Gallery
The gallery figures of the pyrfloc package can be found at
    https://github.com/chenyk1990/gallery/tree/main/pyrfloc
Each figure in the gallery directory corresponds to a DEMO script in the "demo" directory with the exactly the same file name.

The following figure shows the location result of the Texas M4.9 Mentone earthquake (texnet2020galz). Catalog location: catalog (Red): -104.05 31.7 7.1; RFloc3D result (Green): -104.02423355 31.68684031 7.83275 
<img src='https://github.com/chenyk1990/gallery/blob/main/pyrfloc/first_texnet2020galz.png' alt='Slicing' width=960/>

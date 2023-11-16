# Colori-DT

A tool for color difference measurement and neural style transfer between images. This tool has  been translated from its [MATLAB predecessor](https://sourceforge.net/projects/colori-dt/), minus the  Neural Style Transfer function.

To run the tool it is required to have installed Python=> 3.9, preferably in a venv.

The code has been tested with Python 3.9 and 3.10 on MacOS Ventura 13.4.1 and with Python 3.9 on Ubuntu 20.20 

## Features

* Color difference tool: this tool allows for the calculation of point-wise color difference in images with matching size. It implements several color difference metrics: 
    - RGB euclidean
    - CIEDE(LAB) 1976
    - CIEDE(LAB) 1994
    - CIEDE(LAB) 2000
    - CIEDE(LUV) 1976
    - DE-CMC(l:c)(LUV)
    - ICSMC (LUV)
All the implementation are based on previous MATLAB implementations by the Author. 
* Neural style transfer tool: this tool is an implementation of an algorithm that allows for transfers the style of an image onto the content of a second image. To learn more: [Leon et al.:  A neural Algorithm of Artistic Style](https://doi.org/10.48550/arXiv.1508.06576)
**NOTE** To run the neural style transfer, it is stronlgy suggested to have use a GPU, otherwise the process can either hung or run indefinitely.



## Installation 
Install the requirements with:
```
pip install -r requirements.txt
```
or, if on MacOS,
```
pip install -r requirements_macos.txt

```

## Run the tool
To run the Colori-DT GUI, activate the venv, if used and run:
```
python main.py
```


## Run the tests
To run the test it is required to have installed Pytest.
Run the test in the root folder of the project executing the command:
```
pytest
```

## Known bugs
* The metrics CIEDE (LAB) 1994, 2000, and DE-CMC(l:c)(LAB) might raise errors in the terminal where the tool is launched due to operations with large integers and large rational numbers.
* Some .tiff images are wronglydisplayedin the preview windows of the tool. The preview does not affect the calculations, that are carried out correctly

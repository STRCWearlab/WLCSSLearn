# WLCSSLearn   
 
Training algorithm for Warping Longest Common-Subsequence based on GA.

## Requirements:
- Python 3
- CUDA
- progressbar (Be sure to select the package python-progressbar 3.53.*, as there are multiple python packages named progressbar)

This code has been tested on Linux, but with the right adjustment should work on Windows.
Since NVIDIA dropped support for CUDA on macOS, we cannot guarantee that this code works under macOS. 

## Setup

Before to use the code, the CUDA libraries must be compiled.
1) Open a terminal
2) Move to ```libs/cuda/training/params```
3) Execute ```compile_libraries.sh```

## Run

The code offers a ```GAParamsOptimizer``` object that must be initialised.
To start the training, launcher the method ```optimize()```.
The results can be retrieved by launching ```get_results()```.

## References:

The usage of this application in research and publications must be acknowledged by citing the following publication:

[1] Mathias Ciliberto, Luis Ponce Cuspinera, and Daniel Roggen. *"WLCSSLearn: learning algorithm for template matching-based gesture recognition systems."* International Conference on Activity and Behavior Computing. Institute of Electrical and Electronics Engineers, 2019.

```
@inproceedings{Ciliberto2019,
  doi = {10.1109/iciev.2019.8858539},
  url = {https://doi.org/10.1109/iciev.2019.8858539},
  year = {2019},
  month = may,
  publisher = {{IEEE}},
  author = {Mathias Ciliberto and Luis Ponce Cuspinera and Daniel Roggen},
  title = {{WLCSSLearn}: Learning Algorithm for Template Matching-based Gesture Recognition Systems},
  booktitle = {2019 Joint 8th International Conference on Informatics,  Electronics {\&} Vision ({ICIEV}) and 2019 3rd International Conference on Imaging,  Vision {\&} Pattern Recognition ({icIVPR})}
}
```
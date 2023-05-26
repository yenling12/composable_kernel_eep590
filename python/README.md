# CK Python API
This API uses Python to generate instances of operations present in CK, compiles them into a shared library, and an executable to run the instances.
There are 2 directories: shared and normal. The normal directory contains one instance that will compile into an excutable to be run, while the shared directory 
generates multiple instances and compiles them into a shared library.

## Normal
To generate the cpp file and executable:  

`python3 gemm_ex.py`

Run the executable (same layout as CK examples: verification, initialization, run kernel # of times):  

`./ex 0 1 5`

## Shared
Generate all instances, make shared library and executable:  

`python3 driver.py`

Run the executable (same layout as CK examples):  

`./example 0 1 5`

* There's a main.cpp file needed for the executable included, so be careful when deleting the generated cpp files for the instances

The design for parts of this code was taken from Meta's AIT library


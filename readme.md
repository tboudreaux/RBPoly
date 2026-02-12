# FEM rotating polytrope

A model of a rotating barotropic fluid including envelope deformation

## Building
In order to build this you will need both MFEM and CLI11 installed on your system, follow the instructions on their respective websites to do so. Once you have those installed, you can build this project using GNU make.

```bash
make
```

## Running
To run the program, you can use the following command:

```bash
./run --help
```

select your desired options and run. If you wish to visualize the output you must have GLVis running on your local machine 
at port 19916.

## Notes
This is a very simple solver, though the physics should be mostly correct in the barotropic case. The primary issue
with this solver is the convergence rate. We have built this as a feasibility study and therefore no effort has been 
made to optimize the solver. Specifically we use a seriese of nested picard iterations to solve the nonlinear system. These do
seem to converge reliably; however, they do so very slowly. 


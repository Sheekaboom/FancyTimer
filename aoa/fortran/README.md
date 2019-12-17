# README FILE FOR USING FORTRAN CODE

## Building with CMAKE
The following steps must first be taken when building on any new system configuration
  1. Delete any current files in the `<project_base>/fortran/build/cmake` directory
  2. change into the build directory with `cd <project_base>/fortran/build/cmake` 

### Building on Windows
This assumes the two steps under 'Building with CMAKE' have been done and the user is in `<project_base>/fortran/build/cmake`
#### With MinGW    
  1. build the project with MinGW and GCC compilers with `cmake -G "MinGW Makefiles" -D CMAKE_C_COMPILER=gcc CMAKE_CXX_COMPILER=g++ ..`
#### With Visual Studio

### Building on Linux
This assumes the two steps under 'Building with CMAKE' have been done and the user is in `<project_base>/fortran/build`
#### With GCC
  1. Run `cmake ..`. This should automatically detect gcc compilers and build the file


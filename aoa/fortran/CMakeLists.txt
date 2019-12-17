cmake_minimum_required(VERSION 3.5)

project(beamforming)
enable_language(Fortran)
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fopenmp -ffree-form -std=f2008 -fimplicit-none -O3")
#set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS}  -fexternal-blas") #this can allow us to link with mkl
#
# Compile.
#
set(test_sources beamforming_generic.f90 beamforming_test.f90 beamforming_serial.f90)
set(test_sources ${test_sources} fortran_support.f90)
set(dependent_libs beamforming_serial.f90 beamforming_generic.f90 fortran_support.f90)
set(mkl_lib_paths C:/ProgramData/Anaconda3/Library/bin/mkl_core.dll)
set(mkl_lib_paths ${mkl_lib_paths} C:/ProgramData/Anaconda3/Library/bin/mkl_intel_lp64.dll)
set(mkl_lib_paths ${mkl_lib_paths} C:/ProgramData/Anaconda3/Library/bin/mkl_sequential.dll)
add_library(beamform_serial SHARED beamforming_serial_c.f90 ${dependent_libs})
#target_compile_options(beamform_serial PUBLIC -Wno-undef)
#add_executable(beamform_test ${test_sources})

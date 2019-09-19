cmake_minimum_required(VERSION 3.5)

project(beamforming)
enable_language(Fortran)

set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -ffree-form -std=f2008 -fimplicit-none -O3")

#
# Compile.
#
set(test_sources beamforming_generic.f90 beamforming_test.f90 beamforming_serial.f90)
set(test_sources ${test_sources} fortran_support.f90)
set(dependent_libs beamforming_serial.f90 beamforming_generic.f90 fortran_support.f90)
add_library(beamform_serial SHARED beamforming_serial_c.f90 ${dependent_libs})
#target_compile_options(beamform_serial PUBLIC -Wno-undef)
#add_executable(beamform_test ${test_sources})

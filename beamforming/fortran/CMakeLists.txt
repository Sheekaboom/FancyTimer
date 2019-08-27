cmake_minimum_required(VERSION 3.5)

project(beamforming)
enable_language(Fortran)

#if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
#    set(dialect "-ffree-form -std=f2008 -fimplicit-none")
#    set(bounds "-fbounds-check")
#endif()
#if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
#    set(dialect "-stand f08 -free -implicitnone")
#    set(bounds "-check bounds")
#endif()
#if(CMAKE_Fortran_COMPILER_ID MATCHES "PGI")
#    set(dialect "-Mfreeform -Mdclchk -Mstandard -Mallocatable=03")
#    set(bounds "-C")
#endif()

#set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} ${bounds}")
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -ffree-form -std=f2008 -fimplicit-none")

#
# Compile.
#
#set(sources beamforming_generic.f beamforming_serial.f)
set(test_sources beamforming_generic.f90 beamforming_test.f90 beamforming_serial.f90)
set(test_sources ${test_sources} fortran_support.f90)
#file(GLOB_RECURSE sources  src/*.f90 src/*.h)
add_executable(beamform_test ${test_sources})

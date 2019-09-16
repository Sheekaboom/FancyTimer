!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! @author Alec Weiss
! @date 8/2019
! @brief This file contains a serial fortran
!   beamforming implementation.
!   remember the equation is given by:
!   A = SUM(W(position)*exp(dot(k,position)))
!   Here lets look at if we were splitting up
!   our beamforming by angle
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

MODULE BEAMFORMING_SERIAL
    !@brief module for defining serial beamforming
    USE beamforming_generic, ONLY: get_k
    IMPLICIT NONE

    CONTAINS 
    SUBROUTINE get_k_vectors(frequency,positions,err_stat)
        ! @brief create a set of k vectors from cartesian positions
        ! @param[in] frequency - frequency to generate 
        ! @param[in] positions - an Nx3 array containing the positions of each element
        ! @param[out] err_stat - output error status
        REAL, intent(in)     :: frequency,positions
        INTEGER, intent(out) :: err_stat

    END

    FUNCTION get_partial_k_vector_azel(az,el)
        ! @brief create a set of partial k vector (e.g. sin(theta)*cos(phi)) to be later
        !   multiplied by k which will then be dotted with our position (x,y,z)
        ! @note THIS IS CURRENTLY NOT USED
        ! @param[in] az - azimuth values
        ! @param[in] el - of elevation value 
        ! @param[out] partial_k_vector - array of shape (3,) (empty array should be passed in) with the partial k vectors for xyz
        REAL, intent(in)                    :: az,el
        REAL, DIMENSION(3)   :: get_partial_k_vector_azel
        get_partial_k_vector_azel(1) = sin(az)*cos(el) ! x
        get_partial_k_vector_azel(2) = sin(el)         ! y
        get_partial_k_vector_azel(3) = cos(az)*cos(el) ! z
    END

    SUBROUTINE get_steering_vector(frequency,positions,az,el,steering_vectors)
        ! @brief get our steering vectors at a given frequency, positinos, angles
        ! @param[in] frequency - frequency we are beamforming at
        ! @param[in] positions - Nx3 array of positions ((:,c) c=1,2,3 are x,y,z respectively
        ! @param[in] az - azimuth to get the steering vector for
        ! @param[in] el - elevation to calculate the steering vector for
        ! @param[in] steering_vectors - empty complex array of length N to be filled for the steering vvectors of each element
        ! @note this can also be used to generate synthetic plane waves
        REAL, intent(in) :: frequency, az, el
        REAL, intent(in), DIMENSION(:,:) :: positions
        COMPLEX, intent(out), DIMENSION(:) :: steering_vectors
        REAL,DIMENSION(3) :: pkv,k !partial k vector,k value
    END

    COMPLEX FUNCTION get_beamformed_value_c(frequency,positions,weights,meas_vals,az,el)
        ! @brief get a beamformed value at a given freq, positions, and angles
        ! @param[in] frequency - frequency we are beamforming at
        ! @param[in] positions - Nx3 array of positions ((:,c) c=1,2,3 are x,y,z respectively
        ! @param[in] weights - complex weights to apply to each of the elements (should be vector of length N)
        ! @param[in] meas_vals - complex measurements each of the elements (should be vector of length N)
        ! @param[in] az - azimuth to beamform at
        ! @param[in] el - elevation to beamform at
        REAL, intent(in) :: frequency, az, el
        REAL, intent(in), DIMENSION(:,:) :: positions
        COMPLEX, intent(in), DIMENSION(:) :: weights, meas_vals
        INTEGER, DIMENSION(2) :: num_positions
        COMPLEX,ALLOCATABLE, DIMENSION(:) :: steering_vectors
    END FUNCTION


END MODULE BEAMFORMING_SERIAL





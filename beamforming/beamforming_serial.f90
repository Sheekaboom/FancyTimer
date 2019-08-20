!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! @author Alec Weiss
! @date 8/2019
! @brief This file contains a serial fortran
!   beamforming implementation.
!   remember the equation is given by:
!   A = SUM(W(position)*exp(dot(k,position)))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

MODULE BEAMFORMING_SERIAL
    !@brief module for defining serial beamforming
    USE beamforming_generic, ONLY: get_k
    IMPLICIT NONE

    CONTAINS 
    SUBROUTINE generate_k_vectors(frequency,positions,err_stat)
        ! @brief create a set of k vectors from cartesian positions
        ! @param[in] frequency - frequency to generate 
        ! @param[in] positions - an Nx3 array containing the positions of each element
        ! @param[out] err_stat - output error status
        REAL, intent(in)     :: frequency,positions
        INTEGER, intent(out) :: err_stat
    END

    SUBROUTINE generate_partial_k_vector(az_list,el_list,partial_k_vectors,err_stat)
        ! @brief create a set of partial k vector (e.g. sin(theta)*cos(phi)) to be later
        !   multiplied by k which will then be dotted with our position (x,y,z)
        ! @note THIS IS CURRENTLY NOT USED
        ! @param[in] az_list - list of azimuth values with length N
        ! @param[in] el_vals - list of elevation values with length N
        ! @param[out] partial_k_vectors - array of shape Nx3 (empty array should be passed in) with the partial k vectors
        ! @param[out] err_stat - output error status
        REAL, intent(in), DIMENSION(:)      :: az_list,el_list
        REAL, intent(inout), DIMENSION(:,:) :: partial_k_vectors 
        INTEGER, intent(out)                :: err_stat 
        INTEGER, DIMENSION(2)               :: partial_k_shape !shape of partial k value
        partial_k_shape = shape(partial_k_vectors)
        IF(size(az_list)/=size(el_list)) then !ensure sizes are good
            err_stat = -1 
            RETURN
        ELSE IF(partial_k_shape(1)/=size(az_list)) then
            err_stat = -2
            RETURN
        ELSE IF(partial_k_shape(2)/=3) then
            err_stat = -3
            RETURN
        ENDIF
        partial_k_vectors(:,1) = sin(az_list)*cos(el_list)
        partial_k_vectors(:,2) = cos(az_list)*sin(el_list)
        partial_k_vectors(:,3) = cos(az_list)*cos(el_list)
    END

END MODULE BEAMFORMING_SERIAL





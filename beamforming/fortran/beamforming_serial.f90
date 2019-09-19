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
    !USE iso_c_bindings
    IMPLICIT NONE


    CONTAINS 

    FUNCTION get_k_vector_azel(freq,az,el)
        ! @brief create a set of k vectors (e.g. sin(theta)*cos(phi) - k_x,k_y,k_z) 
        !   for a given frequency will then be dotted with our position (x,y,z)
        ! @note THIS IS CURRENTLY NOT USED
        ! @param[in] az - azimuth values
        ! @param[in] el - of elevation value 
        ! @param[out] partial_k_vector - array of shape (3,) (empty array should be passed in) with the partial k vectors for xyz
        REAL, intent(in)                    :: az,el,freq
        REAL                                :: k
        REAL,              DIMENSION(3)     :: get_k_vector_azel
        k = get_k(freq,1.,1.)
        get_k_vector_azel(1) = sin(az)*cos(el) ! x
        get_k_vector_azel(2) = sin(el)         ! y
        get_k_vector_azel(3) = cos(az)*cos(el) ! z
        get_k_vector_azel = k*get_k_vector_azel
    END

    SUBROUTINE get_steering_vectors_(frequency,positions,az,el,steering_vectors)
        ! @brief get our steering vectors at a given frequency, positinos, angles. This will be wrapped by a iso_c function
        ! @param[in] frequency - frequency we are beamforming at
        ! @param[in] positions - Nx3 array of positions ((:,c) c=1,2,3 are x,y,z respectively
        ! @param[in] az - azimuth to get the steering vector for
        ! @param[in] el - elevation to calculate the steering vector for
        ! @param[in] steering_vectors - empty complex array of shape (size(positions)(1),size(az))
        ! @note this can also be used to generate synthetic plane waves
        REAL,   intent(in)                  :: frequency
        REAL,   intent(in), DIMENSION(:)    :: az,el
        REAL,   intent(in), DIMENSION(:,:)  :: positions
        COMPLEX,intent(out),DIMENSION(:,:)  :: steering_vectors
        REAL,               DIMENSION(3)    :: kvec !k vector for a given angle
        INTEGER :: i
        DO i=1,SIZE(az)
            kvec = get_k_vector_azel(frequency,az(i),el(i))
            steering_vectors(:,i) = exp(CMPLX(0,-1)*matmul(positions,kvec)) ! e^(-j*dot(k,r))
        ENDDO
    END

    SUBROUTINE get_beamformed_values_(freqs,positions,weights,meas_vals,az,el,out_vals)
        ! @brief get a beamformed value at a given freq, positions, and angles
        ! @param[in] frequency - frequency we are beamforming at
        ! @param[in] positions - Nx3 array of positions ((:,c) c=1,2,3 are x,y,z respectively
        ! @param[in] weights - complex weights to apply to each of the elements (should be vector of length N)
        ! @param[in] meas_vals - complex measurements each of the elements (should be vector of length N)
        ! @param[in] az - azimuth to beamform at
        ! @param[in] el - elevation to beamform at
        ! @param[in] out_vals - output value matrix (:,n) is each frequency (n,:) is each az/el pair
        REAL,    intent(in), DIMENSION(:)   :: freqs, az, el
        REAL,    intent(in), DIMENSION(:,:) :: positions
        COMPLEX, intent(in), DIMENSION(:)   :: weights 
        COMPLEX, intent(in), DIMENSION(:,:) :: meas_vals ! meas_vals(:,n) is each frequency (n,:) is each az/el pair
        INTEGER,             DIMENSION(2)   :: num_positions
        COMPLEX,ALLOCATABLE, DIMENSION(:,:) :: steering_vectors
        COMPLEX,intent(out), DIMENSION(:,:) :: out_vals !oru output values
        INTEGER                             :: fn,an
        num_positions = shape(positions)
        allocate(steering_vectors(num_positions(1),size(az))) ! we need a steering vector for each xyz position triplet
        DO fn=1,SIZE(freqs)
            call get_steering_vectors_(freqs(fn),positions,az,el,steering_vectors) !get the steering vectors
            DO an=1,size(az)
                out_vals(an,fn) = SUM(weights*meas_vals(:,fn)*steering_vectors(:,an))
            ENDDO
        ENDDO
        deallocate(steering_vectors)
    END SUBROUTINE

    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!! FORTRAN PYTHON INTERFACING TESTS
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    FUNCTION fortran_mult_funct(test_num_a,test_num_b)
        ! @brief test python with fortran functions
        REAL, intent(in) :: test_num_a,test_num_b
        REAL :: fortran_mult_funct 
        fortran_mult_funct= test_num_a*test_num_b
        RETURN
    END FUNCTION

    SUBROUTINE fortran_array_mult_sub(arr_a,arr_b,out_arr,n)
        ! @brief test python with fortran subroutines
        INTEGER, intent(in) :: n
        REAL, intent(in), DIMENSION(n) :: arr_a,arr_b
        REAL, intent(inout), DIMENSION(n) :: out_arr 
        out_arr = arr_a*arr_b
    END SUBROUTINE

    SUBROUTINE fortran_array_math_sub(arr_a,arr_b,out_arr,n)
        ! @brief test python with fortran subroutines
        INTEGER, intent(in) :: n
        REAL, intent(in), DIMENSION(n) :: arr_a,arr_b
        REAL, intent(inout), DIMENSION(n) :: out_arr 
        out_arr = arr_a*arr_b
        out_arr = arr_a+out_arr 
        out_arr = arr_b-out_arr 
        out_arr = out_arr*(arr_a/arr_b)
    END SUBROUTINE

END MODULE BEAMFORMING_SERIAL





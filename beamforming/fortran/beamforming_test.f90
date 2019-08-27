!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! @author Alec Weiss
! @date 8/2019
! @brief This file tests beamforming code
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

PROGRAM BEAMFORM_TEST

USE beamforming_generic, ONLY: get_k,get_lambda,deg2rad,rad2deg
USE beamforming_serial, ONLY: get_partial_k_vector_azel,get_steering_vector,get_beamformed_value
USE fortran_support, ONLY: print_matrix
IMPLICIT None

    !!test getting partial k vectors
    !REAL :: lam,k,frequency
    !REAL :: test
    !REAL :: az_val, el_val
    !REAL, DIMENSION(3) :: partial_k_vec
    !INTEGER :: n,err_stat
    !az_val = 0; el_val = 0
    !partial_k_vec = get_partial_k_vector_azel(az_val,el_val)
    !print*, partial_k_vec
    
    !! test generic functions
    !frequency = 40e9; lam = 0; k=0
    !k = get_k(frequency,1.,1.)
    !lam = get_lambda(frequency,1.,1.)
    !print*, "Lambda = ",lam,"  K = ",k
    !print*, "AZ: ",az_vals, "   EL:",el_vals
    !print*, "AZ: ",deg2rad(az_vals), "   EL:",deg2rad(el_vals)
    !print*, "AZ: ",cos(deg2rad(az_vals)), "   EL:",cos(deg2rad(el_vals))
    !print*, shape(az_vals)(1)

    !! test steering vectors and beamforming
    REAL :: frequency = 40e9, lam_2
    REAL :: az = 45., el = 0.
    INTEGER :: x_grid_size = 35,y_grid_size = 35,num_positions
    INTEGER :: num_az = 256, num_el = 256
    REAL, ALLOCATABLE,DIMENSION(:) :: az_vals, el_vals
    INTEGER :: i,j,k,idx
    INTEGER :: err_code
    REAL, ALLOCATABLE, DIMENSION(:,:) :: positions
    COMPLEX, ALLOCATABLE, DIMENSION(:) :: steering_vectors, array_weights, array_data
    COMPLEX, ALLOCATABLE, DIMENSION(:,:) :: beamformed_data
    !first calculate our positions
    num_positions = x_grid_size*y_grid_size
    ALLOCATE(positions(num_positions,3),steering_vectors(num_positions))
    positions = 0
    print*,'Number of Positions',num_positions
    lam_2 = get_lambda(frequency,1.,1.)/2. !calculate lamdba/2
    print*,'Lambda/2 = ',lam_2
    DO i=0,x_grid_size-1
        DO j=0,y_grid_size-1
                k = 0
                idx = (i+(j*x_grid_size))+1
                positions(idx,1) = lam_2*i !x positions
                positions(idx,2) = lam_2*j !y positions
                !print*, idx
        END DO
    END DO 
    !err_code = print_matrix(positions) !check positions
    !err_code = print_matrix(steering_vectors)
    !call get_steering_vector(frequency,positions,az,el,steering_vectors)
    !err_code = print_matrix(steering_vectors)
    !! now for beamforming
    allocate(beamformed_data(num_az,num_el),az_vals(num_az),el_vals(num_el))
    allocate(array_weights(num_positions),array_data(num_positions))
    array_weights = CMPLX(1,0); array_data = CMPLX(1,0)
    DO i=1,num_az !populate azimuth
        az_vals(i) = i*(num_az/180.)
    END DO
    DO i=1,num_el !populate elevation
        el_vals(i) = i*(num_el/180.)
    END DO
    !now loop through and beamform
    DO i=1,num_az !loop az
        az = az_vals(i)
        DO j=1,num_el !loop el
            el = el_vals(i)
            beamformed_data(i,j) = get_beamformed_value(frequency,positions,array_weights,array_data,az,el)
        END DO 
    END DO 
    !err_code = print_matrix(beamformed_data)

END PROGRAM BEAMFORM_TEST






!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! @author Alec Weiss
! @date 8/2019
! @brief This file tests beamforming code
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

PROGRAM BEAMFORM_TEST

USE beamforming_generic, ONLY: get_k,get_lambda,deg2rad,rad2deg
USE beamforming_serial, ONLY: generate_partial_k_vectors_azel
IMPLICIT None

    REAL :: lam,k,frequency
    REAL :: test
    REAL, DIMENSION(10) :: az_vals, el_vals
    REAL, DIMENSION(10,3) :: partial_k_vecs
    INTEGER :: n,err_stat
    do n=1,10
        az_vals(n) = n*9
        el_vals(n) = 90-(n*9)
    end do
    call generate_partial_k_vectors_azel(az_vals,el_vals,partial_k_vecs,err_stat)
    print*, partial_k_vecs
    !frequency = 40e9; lam = 0; k=0
    !k = get_k(frequency,1.,1.)
    !lam = get_lambda(frequency,1.,1.)
    !print*, "Lambda = ",lam,"  K = ",k
    !print*, "AZ: ",az_vals, "   EL:",el_vals
    !print*, "AZ: ",deg2rad(az_vals), "   EL:",deg2rad(el_vals)
    !print*, "AZ: ",cos(deg2rad(az_vals)), "   EL:",cos(deg2rad(el_vals))
    !print*, shape(az_vals)(1)


END PROGRAM BEAMFORM_TEST






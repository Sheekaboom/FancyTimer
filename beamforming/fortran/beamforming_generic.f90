!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! @author Alec Weiss
! @date 8/2019
! @brief This file covers generic constants
!   and functions to be used in beamforming
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

MODULE BEAMFORMING_GENERIC
    !define some constants here
    REAL :: SPEED_OF_LIGHT = 299792458.0 !speed of light in vacuum
    REAL :: PI = 3.141592654
    
    Contains
        REAL FUNCTION get_k(frequency,eps_r,mu_r)
            ! @brief get the wavenumber from given parameters
            ! @param[in] frequency - frequency to get the wavenumber for
            ! @param[in] eps_r - relative permittivity (usually 1)
            ! @param[in] mu_r - relative permeability (usually 1)
            ! @return wavenumber from the provided input parameters
            REAL, intent(in) :: frequency
            REAL, intent(in):: eps_r, mu_r
            REAL :: lambda
            !check if eps_r and mu_r are provided
            lambda = get_lambda(frequency,eps_r,mu_r) 
            get_k = 2*PI/lambda
            RETURN
        END

        REAL FUNCTION get_lambda(frequency,eps_r,mu_r)
            ! @brief get the wavelength from a given frequency
            ! @param[in] frequency - frequency to get the wavenumber for
            ! @param[in] eps_r - relative permittivity (usually 1)
            ! @param[in] mu_r - relative permeability (usually 1)
            ! @return wavelength from the provided input parameters
            REAL, intent(in) :: frequency
            REAL, intent(in) :: eps_r, mu_r
            !check if eps_r and mu_r are provided
            get_lambda = SPEED_OF_LIGHT/sqrt(eps_r*mu_r)/frequency
            RETURN
        END

        ELEMENTAL REAL FUNCTION deg2rad(angle_degrees)
            ! @brief change degrees to radians
            ! @param[in] angle_degrees - value in degrees to change to radians 
            ! return angle in radians
            REAL, intent(in) :: angle_degrees
            deg2rad = angle_degrees*PI/180.
            RETURN
        END

        ELEMENTAL REAL FUNCTION rad2deg(angle_radians)
            ! @brief change degrees to radians
            ! @param[in] angle_radians - value in radians to change to degrees
            ! return angle in degrees
            REAL, intent(in) :: angle_radians
            rad2deg = angle_radians*180./PI
            RETURN
        END

        ELEMENTAL REAL FUNCTION lin2db(val_linear)
            ! @brief change a linear value to db (10*log10 value)
            REAL, intent(in) :: val_linear
            lin2db = 10*LOG10(val_linear)
            RETURN
        END

        ELEMENTAL REAL FUNCTION angle(complex_value)
            ! @brief compute the angle of a complex number
            COMPLEX, intent(in) :: complex_value
            angle = ATAN2(AIMAG(complex_value),REAL(complex_value))
            RETURN
        END

END MODULE BEAMFORMING_GENERIC
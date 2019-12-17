!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! @author Alec Weiss
! @date 8/2019
! @brief This file contains helper functions
!  for various fortran things
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

MODULE FORTRAN_SUPPORT

    IMPLICIT NONE
    INTEGER, PARAMETER :: DP=8

    INTERFACE print_matrix
        MODULE PROCEDURE print_matrix_REAL
        MODULE PROCEDURE print_matrix_COMPLEX
        MODULE PROCEDURE print_vector_REAL
        MODULE PROCEDURE print_vector_COMPLEX
    END INTERFACE

    CONTAINS


        INTEGER FUNCTION print_matrix_REAL(my_matrix)
            ! @brief function to print a matrix in a more readable manner (2D max)
            ! @param[in] my_matrix - matrix to print
            REAL(DP), intent(in), DIMENSION(:,:) :: my_matrix
            INTEGER, DIMENSION(2) :: matrix_shape
            INTEGER :: i
            matrix_shape = shape(my_matrix)
            do i=1,matrix_shape(1)
                print *, my_matrix(i,:)
            end do
            print_matrix_REAL=0 !success
            RETURN
        END
        INTEGER FUNCTION print_matrix_COMPLEX(my_matrix)
            ! @brief function to print a complex matrix in a more readable manner (2D max)
            ! @param[in] my_matrix - matrix to print
            COMPLEX(DP), intent(in), DIMENSION(:,:) :: my_matrix
            INTEGER, DIMENSION(2) :: matrix_shape
            INTEGER :: i
            matrix_shape = shape(my_matrix)
            do i=1,matrix_shape(1)
                print *, my_matrix(i,:)
            end do
            print_matrix_COMPLEX=0 !success
            RETURN
        END
        INTEGER FUNCTION print_vector_REAL(my_matrix)
            ! @brief function to print a matrix in a more readable manner (2D max)
            ! @param[in] my_matrix - matrix to print
            REAL(DP), intent(in), DIMENSION(:) :: my_matrix
            INTEGER, DIMENSION(1) :: matrix_shape
            INTEGER :: i
            matrix_shape = shape(my_matrix)
            do i=1,matrix_shape(1)
                print *, my_matrix(i)
            end do
            print_vector_REAL=0 !success
            RETURN
        END
        INTEGER FUNCTION print_vector_COMPLEX(my_matrix)
            ! @brief function to print a complex matrix in a more readable manner (2D max)
            ! @param[in] my_matrix - matrix to print
            COMPLEX(DP), intent(in), DIMENSION(:) :: my_matrix
            INTEGER, DIMENSION(1) :: matrix_shape
            INTEGER :: i
            matrix_shape = shape(my_matrix)
            do i=1,matrix_shape(1)
                print *, my_matrix(i)
            end do
            print_vector_COMPLEX=0 !success
            RETURN
        END

END MODULE FORTRAN_SUPPORT


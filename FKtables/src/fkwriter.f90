MODULE fkwriter

   USE kinds

   IMPLICIT NONE

   !TYPE(pineappl_lumi) :: lumi
   !TYPE(pineappl_grid) :: grid
   !TYPE(pineappl_keyval) :: key_vals

   CONTAINS

      SUBROUTINE write_to_txt(xnu,wgt,filename)
         REAL(KIND=dp),DIMENSION(:),INTENT(IN) :: xnu
         REAL(KIND=dp),DIMENSION(:,:),INTENT(IN) :: wgt
         CHARACTER(LEN=*),INTENT(IN) :: filename

         INTEGER :: iunit,i,info

         OPEN(NEWUNIT=iunit,FILE=TRIM(filename),ACTION="WRITE",IOSTAT=info)
         WRITE(iunit,*) xnu
         WRITE(*,"(A)") "--------"
         WRITE(*, *) "wgt shape: ", SIZE(wgt, 1), SIZE(wgt, 2)
         DO i=LBOUND(wgt,2),UBOUND(wgt,2)
            print *, i
            PRINT *, wgt(:, i)
            WRITE(iunit, '(50E25.16)') wgt(:, i)
         END DO
      CLOSE(iunit)
   
         ! WRITE(*, *) "wgt shape: ", SIZE(wgt, 1), SIZE(wgt, 2)
         !        WRITE(iunit,'(E25.16)') wgt(:,i)
         ! END DO
      END SUBROUTINE write_to_txt



END MODULE fkwriter



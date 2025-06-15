PROGRAM genfk

   USE observables
   USE structure_functions
   USE fkwriter

   IMPLICIT NONE

   INTEGER,PARAMETER :: NARR=50 ! Number of grid points
   INTEGER,PARAMETER :: NBIN=100 ! Number of bins in the FKtable


   REAL(KIND=dp),DIMENSION(NARR) :: arr,farr,res
   REAL(KIND=dp),DIMENSION(NARR,NBIN) :: FK ! FK(NARR,LUMI=1,NBIN,1) 
                                            ! therefore only 2d-array
   REAL(KIND=dp) :: N
   INTEGER(KIND=4) :: i,j
   REAL(KIND=dp),DIMENSION(NBIN+1) :: bins
   REAL(KIND=dp) :: binwidth
   REAL(KIND=dp),DIMENSION(NBIN) :: Enu
   INTEGER(KIND=4),DIMENSION(2) :: pdg_id_pairs
   REAL(KIND=dp),DIMENSION(1) :: factors

   CHARACTER(:), ALLOCATABLE :: filename

   filename = "fk_table_100"
   ! Initialise the interpolation grid
   CALL logspaced_grid(NARR,-5,arr)
   CALL init_grid_and_interpolation(arr)

   ! Parameters for the FKtable
   ! Lumi entry in the FKtable.
   pdg_id_pairs=(/12,12/)
   factors=(/1.0_dp/)
   ! Set up the lower and uppper bounds of the FKtable bins.
   DO i=LBOUND(bins,1),UBOUND(bins,1)
      bins(i)=25.0_dp+(REAL(i-1,KIND=dp)*(6000.0_dp-25.0_dp) &
                       /REAL(NBIN,KIND=dp))
   END DO
   ! Compute the Energy at which the FKtable entries are computed.
   ! i.e. the mid of the bin.
   DO i=2,UBOUND(bins,1)
      Enu(i-1)=(bins(i)+bins(i-1))/2.0_dp
      !Enu(i-1)=bins(i)
   END DO
   

   ! Use the NNSFnu structure functions.
   CALL init_pdfs("NNSFnu_W_highQ",1)
   CALL init_pdfs("faserv",2)

   FK=0.0_dp
   DO i=LBOUND(FK,1),UBOUND(FK,1)
      DO j=LBOUND(FK,2),UBOUND(FK,2)
         CALL compute_dNdEnu(Enu(j),i,FK(i,j))
      END DO
   END DO

   CALL write_to_txt(arr,FK,filename)

END PROGRAM genfk

PROGRAM FKtable_nuflux

   USE kinds
   USE interpolation
   USE observables
!   USE quadpack, ONLY: DQAG

   IMPLICIT NONE
   TYPE(interpolation_grid) :: xgrid
   REAL(KIND=dp),DIMENSION(41) :: arr,farr,res
   REAL(KIND=dp) :: r,sum
   INTEGER(KIND=4) :: i,j,alpha
   REAL(KIND=dp) :: xn

   arr=(/ 8.5168066775733548E-006_dp, &  
          1.2921015690747310E-005_dp, &
          1.9602505002391748E-005_dp, &
          2.9738495372244901E-005_dp, &
          4.5114383949640441E-005_dp, &
          6.8437449189678965E-005_dp, &
          1.0381172986576898E-004_dp, &
          1.5745605600841445E-004_dp, &
          2.3878782918561914E-004_dp, &
          3.6205449638139736E-004_dp, &
          5.4877953236707956E-004_dp, &
          8.3140688364881441E-004_dp, &
          1.2586797144272762E-003_dp, &
          1.9034634022867384E-003_dp, &
          2.8738675812817515E-003_dp, &
          4.3285006388208112E-003_dp, &
          6.4962061946337987E-003_dp, &
          9.6991595740433985E-003_dp, &
          1.4375068581090129E-002_dp, &
          2.1089186683787169E-002_dp, &
          3.0521584007828916E-002_dp, &
          4.3414917417022691E-002_dp, &
          6.0480028754447364E-002_dp, &
          8.2281221262048926E-002_dp, &
         0.10914375746330703_dp,&     
         0.14112080644440345_dp,& 
         0.17802566042569432_dp,&     
         0.21950412650038861_dp,&     
         0.26511370415828228_dp,&     
         0.31438740076927585_dp,&     
         0.36687531864822420_dp,&     
         0.42216677535896480_dp,&     
         0.47989890296102550_dp,&     
         0.53975723378804452_dp,&     
         0.60147219796733498_dp,&     
         0.66481394824738227_dp,&     
         0.72958684424143116_dp,&     
         0.79562425229227562_dp,&     
         0.86278393239061080_dp,&     
         0.93094408087175440_dp,&     
          1.0000000000000000_dp/)

   CALL init_grid_and_interpolation(arr)
   xn=2000.0_dp/14000.0_dp
   DO alpha=LBOUND(arr,1),UBOUND(arr,1)
      res(alpha)=fktable_entry(alpha)
   END DO
   
   CALL init_pdfs("faserv")
   DO i=LBOUND(arr,1),UBOUND(arr,1)
      CALL lhapdf_xfxq(1,0,12,arr(i),2.0_dp,farr(i))
      farr(i)=farr(i)/arr(i)
   END DO
   sum=0.0_dp
   DO alpha=LBOUND(arr,1),UBOUND(arr,1)
      WRITE(*,*) farr(alpha),res(alpha),farr(alpha)*res(alpha)
      sum=sum+farr(alpha)*res(alpha)
   END DO
   WRITE(*,*) sum

   


END PROGRAM FKtable_nuflux


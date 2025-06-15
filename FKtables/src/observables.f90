MODULE observables

   USE kinds
   USE structure_functions
   USE interpolation
   USE integration, ONLY: vegas

   IMPLICIT NONE

   !REAL(KIND=dp),PARAMETER :: Mp=0.938_dp ! Mass of the Proton in GeV.
   REAL(KIND=dp),PARAMETER :: Mp=171.25_dp/(110.0_dp*0.93956+74.0_dp*0.93827_dp) ! Mass of the Proton in GeV.
   REAL(KIND=dp),PARAMETER :: EbeamA=14000.0_dp ! The fake energy of the 
   REAL(KIND=dp),PARAMETER :: EbeamB=Mp      
   REAL(KIND=dp),PARAMETER :: GW=0.664_dp    ! Weak coupling constant
   REAL(KIND=dp),PARAMETER :: GW4=GW**4     
   REAL(KIND=dp),PARAMETER :: MW=80.385_dp    ! Mass of the W-boson
   REAL(KIND=dp),PARAMETER :: MW2=MW**2         ! Mass squared of the W-boson.
   REAL(KIND=dp),PARAMETER :: WW=2.095_dp       ! Width of the W-boson.
   !REAL(KIND=dp),PARAMETER :: GF=DSQRT(2.0_dp)/8.0_dp*GW**2/MW2
   REAL(KIND=dp),PARAMETER :: GF=1.1663787E-005_dp

   TYPE(interpolation_grid) :: intrpln

   
   REAL(KIND=dp) :: Enu0,El0
   INTEGER(KIND=4) :: alpha0

   CONTAINS

      SUBROUTINE init_grid_and_interpolation(xgrid)
         REAL(KIND=dp),DIMENSION(:),INTENT(IN) :: xgrid
         ! Do not need to know about the function f(xgrid).
         ! Just need the polynomials for the interpolation 
         ! in log(x).
         intrpln=new_xgrid(xgrid,xgrid)
      END SUBROUTINE init_grid_and_interpolation

      SUBROUTINE get_xQ2yYplusYminusfnujac(par,x,Q2,y,Yplus,Yminus,fnu,jac)
         REAL(KIND=dp),DIMENSION(:),INTENT(IN) :: par
         REAL(KIND=dp),INTENT(OUT) :: x,Q2,jac,fnu,y,Yplus,Yminus
         REAL(KIND=dp) :: s,Enu,xnu
         REAL(KIND=dp) :: Q2max,Q2min,xmin,xmax,ymin,ymax,xnumin,xnumax
         x=par(1)
         Q2=par(2)
         xnu=par(3)

         s=2.0_dp*MP*EbeamA

         ! initial values for the integration boundaries.
         xnumax=1.0_dp
         xnumin=1.0e-9_dp
         xmax=1.0_dp
         xmin=1.0e-5_dp
         Q2min=4.0_dp
         Q2max=s

         IF(xmin*xnumin*s.lt.Q2min) xmin=Q2min/s
         ymin=Q2min/(xmax*xnumax*s)
         ymax=1.0_dp

         x=(xmax-xmin)*par(1)+xmin
         xnumin=Q2min/x/ymax/s
         xnu=(xnumax-xnumin)*par(3)+xnumin
         Enu=xnu*EbeamA
         Q2max=x*xnu*s
         Q2=(Q2max-Q2min)*par(2)+Q2min
         jac=(xmax-xmin)*(Q2max-Q2min)*(xnumax-xnumin)

         y=Q2/(2.0_dp*x*MP*Enu)
         yplus=1.0_dp+(1.0_dp-y)**2
         yminus=1.0_dp-(1.0_dp-y)**2
         CALL lhapdf_xfxQ(2,0,12,xnu,1.0_dp,fnu)
         fnu=fnu/xnu
      END SUBROUTINE get_xQ2yYplusYminusfnujac

      REAL(KIND=dp) FUNCTION d2sigdxdQ2(par,wgt)
         REAL(KIND=dp),DIMENSION(:),INTENT(IN) :: par
         REAL(KIND=dp),INTENT(IN) :: wgt
         REAL(KIND=dp) :: x,Q2,xnu,fnu,jac,s,y,Yplus,Yminus,Enu
         CALL get_xQ2yYplusYminusfnujac(par,x,Q2,y,Yplus,Yminus,fnu,jac)
         d2sigdxdQ2=GF**2/x/4.0_dp/PI/(1.0_dp+Q2/MW2)**2 &
            *(Yplus*F2_nuA(x,Q2)-y**2*FL_nuA(x,Q2)+Yminus*xF3_nuA(x,Q2))*fnu*jac
      END FUNCTION d2sigdxdQ2

      REAL(KIND=dp) FUNCTION d2sigdxdQ2_nb(par,wgt)
         REAL(KIND=dp),DIMENSION(:),INTENT(IN) :: par
         REAL(KIND=dp),INTENT(IN) :: wgt
         REAL(KIND=dp) :: x,Q2,xnu,fnu,jac,s,y,Yplus,Yminus,Enu
         CALL get_xQ2yYplusYminusfnujac(par,x,Q2,y,Yplus,Yminus,fnu,jac)
         d2sigdxdQ2_nb=GF**2/x/4.0_dp/PI/(1.0_dp+Q2/MW2)**2 &
            *(Yplus*F2_nbA(x,Q2)-y**2*FL_nbA(x,Q2)-Yminus*xF3_nbA(x,Q2))*fnu*jac
      END FUNCTION d2sigdxdQ2_nb

      REAL(KIND=dp) FUNCTION d2sigdxdQ2_nun(par,wgt)
         REAL(KIND=dp),DIMENSION(:),INTENT(IN) :: par
         REAL(KIND=dp),INTENT(IN) :: wgt
         REAL(KIND=dp) :: x,Q2,xnu,fnu,jac,s,y,Yplus,Yminus,Enu
         CALL get_xQ2yYplusYminusfnujac(par,x,Q2,y,Yplus,Yminus,fnu,jac)
         d2sigdxdQ2_nun=GF**2/x/4.0_dp/PI/(1.0_dp+Q2/MW2)**2 &
            *(Yplus*F2_nun(x,Q2)-y**2*FL_nun(x,Q2)+Yminus*xF3_nun(x,Q2))*fnu*jac
      END FUNCTION d2sigdxdQ2_nun

      REAL(KIND=dp) FUNCTION d2sigdxdQ2_nbn(par,wgt)
         REAL(KIND=dp),DIMENSION(:),INTENT(IN) :: par
         REAL(KIND=dp),INTENT(IN) :: wgt
         REAL(KIND=dp) :: x,Q2,xnu,fnu,jac,s,y,Yplus,Yminus,Enu
         CALL get_xQ2yYplusYminusfnujac(par,x,Q2,y,Yplus,Yminus,fnu,jac)
         d2sigdxdQ2_nbn=GF**2/x/4.0_dp/PI/(1.0_dp+Q2/MW2)**2 &
            *(Yplus*F2_nbn(x,Q2)-y**2*FL_nbn(x,Q2)+Yminus*xF3_nbn(x,Q2))*fnu*jac
      END FUNCTION d2sigdxdQ2_nbn

      ! Hier nicht mehr ueber xnu integrieren!
      REAL(KIND=dp) FUNCTION d2sigdxdQ2dEnu(par,wgt)
         REAL(KIND=dp),DIMENSION(:),INTENT(IN) :: par
         REAL(KIND=dp),INTENT(IN) :: wgt
         REAL(KIND=dp) :: x,Q2,xnu
         REAL(KIND=dp) :: s,y,Yplus,Yminus,Enu
         REAL(KIND=dp) :: Q2max,Q2min,xmin,xmax
         REAL(KIND=dp) :: fnu,El,Eh,theta
         REAL(KIND=dp) :: jac
         d2sigdxdQ2dEnu=0.0_dp
         x=par(1)
         Q2=par(2)

         s=2.0_dp*MP*EbeamA

         ! initial values for the integration boundaries.
         xmax=1.0_dp
         xmin=1.0e-5_dp
         Q2min=4.0_dp
         xnu=Enu0/EbeamA 
         Enu=Enu0

         IF(xmin*xnu*s.lt.Q2min) xmin=Q2min/(s)

         x=(xmax-xmin)*par(1)+xmin
         Q2max=x*xnu*s
         Q2=(Q2max-Q2min)*par(2)+Q2min

         ! Faserv cuts.
         Eh=Q2/x/MP/2.0_dp
         El=Enu-Eh
         theta=2.0_dp*DASIN(DSQRT(Q2/(4.0_dp*Enu*El)))
         IF(Eh.LT.100.0_dp) RETURN
         IF(El.LT.100.0_dp) RETURN
         IF(DTAN(theta).GT.0.025_dp) RETURN

         y=Q2/(2.0_dp*x*MP*Enu)
         yplus=1.0_dp+(1.0_dp-y)**2
         yminus=1.0_dp-(1.0_dp-y)**2

         jac=(xmax-xmin)*(Q2max-Q2min)/EbeamA
         
         ! Integral kernel
         d2sigdxdQ2dEnu=GF**2/x/4.0_dp/PI/(1.0_dp+Q2/MW2)**2 &
            *(Yplus*F2_nuA(x,Q2)-y**2*FL_nuA(x,Q2)+Yminus*xF3_nuA(x,Q2)) &
            *jac*intrpln%p(xnu,alpha0)
      END FUNCTION d2sigdxdQ2dEnu

      SUBROUTINE compute_dNdEnu(Enu,alpha,dNdEnu)
         REAL(KIND=dp),INTENT(IN) :: Enu
         INTEGER(KIND=4),INTENT(IN) :: alpha
         REAL(KIND=dp),INTENT(OUT) :: dNdEnu

         ! Could also promote some of these to module variables in integration.
         INTEGER(KIND=4) :: init,ncall,itmx,nprn
         REAL(KIND=dp),DIMENSION(4) :: region
         REAL(KIND=dp) :: tgral,sd,chi2a,N1,N2,N3,N4

         REAL(KIND=dp) :: xnu

         dNdEnu=0.0_dp

         itmx=5
         ncall=800000
         nprn=0
         init=0

         region=(/0.0_dp,0.0_dp,1.0_dp,1.0_dp/)
         Enu0=Enu
         xnu=Enu0/EbeamA
         alpha0=alpha
         IF(intrpln%p(xnu,alpha0).LT.EPSILON(1.0_dp)) RETURN
         CALL vegas(region,d2sigdxdQ2dEnu,init,ncall,itmx,nprn,tgral,sd,chi2a)
         dNdEnu=tgral*1E9_dp/2.56819_dp
      END SUBROUTINE compute_dNdEnu

      SUBROUTINE compute_event_yield(N)
         REAL(KIND=dp),INTENT(OUT) :: N
         INTEGER(KIND=4) :: init,ncall,itmx,nprn
         REAL(KIND=dp),DIMENSION(6) :: region
         REAL(KIND=dp) :: tgral,sd,chi2a,N1,N2,N3,N4
         
         itmx=5
         ncall=250000
         nprn=0
         init=0

         region=(/0.0_dp,0.0_dp,0.0_dp,1.0_dp,1.0_dp,1.0_dp/)
         CALL vegas(region,d2sigdxdQ2,init,ncall,itmx,nprn,tgral,sd,chi2a)
         N1=tgral
         !CALL vegas(region,d2sigdxdQ2_nun,init,ncall,itmx,nprn,tgral,sd,chi2a)
         !N2=tgral
         CALL vegas(region,d2sigdxdQ2_nb,init,ncall,itmx,nprn,tgral,sd,chi2a)
         N3=tgral
         !CALL vegas(region,d2sigdxdQ2_nbn,init,ncall,itmx,nprn,tgral,sd,chi2a)
         !N4=tgral
         N2=0.0_dp
         N4=0.0_dp
         N=(183.0_dp-74.0_dp)/183.0_dp*(N2+N4)+74.0_dp/183.0_dp*(N1+N3)
         N=N1+N3
         N=N*1E9_dp/2.56819_dp
      END SUBROUTINE compute_event_yield

END MODULE observables

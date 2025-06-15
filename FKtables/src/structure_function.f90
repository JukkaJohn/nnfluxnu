MODULE structure_functions

   USE kinds

   IMPLICIT NONE

   PUBLIC :: init_pdfs

   CONTAINS

      SUBROUTINE init_pdfs(pdfname,iset)
         IMPLICIT NONE
         CHARACTER(LEN=*),INTENT(IN) :: pdfname
         INTEGER(KIND=4) :: iset
         INTEGER(KIND=4) :: npdf
         CALL lhapdf_initpdfset_byname(iset,ADJUSTL(TRIM(pdfname)))
         CALL numberpdf(npdf)
         WRITE(*,*) "Replicas found: ",npdf
         WRITE(*,*) "PDF set initialized."
      END SUBROUTINE init_pdfs

      REAL(KIND=dp) FUNCTION F2_nuA(x,Q2)
         REAL(KIND=dp),INTENT(IN) :: x,Q2
         CALL lhapdf_xfxq(1,0,1001,x,DSQRT(Q2),F2_nuA)
         F2_nuA=F2_nuA
      END FUNCTION F2_nuA

      REAL(KIND=dp) FUNCTION FL_nuA(x,Q2)
         REAL(KIND=dp),INTENT(IN) :: x,Q2
         CALL lhapdf_xfxq(1,0,1002,x,DSQRT(Q2),FL_nuA)
         FL_nuA=FL_nuA
      END FUNCTION FL_nuA

      REAL(KIND=dp) FUNCTION xF3_nuA(x,Q2)
         REAL(KIND=dp),INTENT(IN) :: x,Q2
         CALL lhapdf_xfxq(1,0,1003,x,DSQRT(Q2),xF3_nuA)
         xF3_nuA=xF3_nuA
      END FUNCTION xF3_nuA

      REAL(KIND=dp) FUNCTION F2_nbA(x,Q2)
         REAL(KIND=dp),INTENT(IN) :: x,Q2
         CALL lhapdf_xfxq(1,0,2001,x,DSQRT(Q2),F2_nbA)
         F2_nbA=F2_nbA
      END FUNCTION F2_nbA

      REAL(KIND=dp) FUNCTION FL_nbA(x,Q2)
         REAL(KIND=dp),INTENT(IN) :: x,Q2
         CALL lhapdf_xfxq(1,0,2002,x,DSQRT(Q2),FL_nbA)
         FL_nbA=FL_nbA
      END FUNCTION FL_nbA

      REAL(KIND=dp) FUNCTION xF3_nbA(x,Q2)
         REAL(KIND=dp),INTENT(IN) :: x,Q2
         CALL lhapdf_xfxq(1,0,2003,x,DSQRT(Q2),xF3_nbA)
         xF3_nbA=xF3_nbA
      END FUNCTION xF3_nbA

      REAL(KIND=dp) FUNCTION f(pid,x,Q2)
         INTEGER(KIND=4),INTENT(IN) :: pid
         REAL(KIND=dp),INTENT(IN) :: x,Q2 
         CALL lhapdf_xfxq(1,0,pid,x,DSQRT(Q2),f)
         f=f/x
      END FUNCTION f

      REAL(KIND=dp) FUNCTION F2_nup(x,Q2)
         REAL(KIND=dp),INTENT(IN) :: x,Q2 
         F2_nup=2.0_dp*x*(f(-2,x,Q2)+f(1,x,Q2)+f(3,x,Q2)+f(-4,x,Q2))
      END FUNCTION F2_nup

      REAL(KIND=dp) FUNCTION FL_nup(x,Q2)
         REAL(KIND=dp),INTENT(IN) :: x,Q2
         FL_nup=0.0_dp
      END FUNCTION FL_nup

      REAL(KIND=dp) FUNCTION xF3_nup(x,Q2)
         REAL(KIND=dp),INTENT(IN) :: x,Q2 
         xF3_nup=2.0_dp*x*(-f(-2,x,Q2)+f(1,x,Q2)+f(3,x,Q2)-f(-4,x,Q2))
      END FUNCTION xF3_nup

      REAL(KIND=dp) FUNCTION F2_nbp(x,Q2)
         REAL(KIND=dp),INTENT(IN) :: x,Q2 
         F2_nbp=2.0_dp*x*(f(2,x,Q2)+f(-1,x,Q2)+f(-3,x,Q2)+f(4,x,Q2))
      END FUNCTION F2_nbp

      REAL(KIND=dp) FUNCTION FL_nbp(x,Q2)
         REAL(KIND=dp),INTENT(IN) :: x,Q2
         FL_nbp=0.0_dp
      END FUNCTION FL_nbp

      REAL(KIND=dp) FUNCTION xF3_nbp(x,Q2)
         REAL(KIND=dp),INTENT(IN) :: x,Q2 
         xF3_nbp=2.0_dp*x*(f(2,x,Q2)-f(-1,x,Q2)-f(-3,x,Q2)+f(4,x,Q2))
      END FUNCTION xF3_nbp

      REAL(KIND=dp) FUNCTION F2_nun(x,Q2)
         REAL(KIND=dp),INTENT(IN) :: x,Q2 
         F2_nun=2.0_dp*x*(f(-1,x,Q2)+f(2,x,Q2)+f(3,x,Q2)+f(-4,x,Q2))
      END FUNCTION F2_nun

      REAL(KIND=dp) FUNCTION FL_nun(x,Q2)
         REAL(KIND=dp),INTENT(IN) :: x,Q2
         FL_nun=0.0_dp
      END FUNCTION FL_nun

      REAL(KIND=dp) FUNCTION xF3_nun(x,Q2)
         REAL(KIND=dp),INTENT(IN) :: x,Q2 
         xF3_nun=2.0_dp*x*(-f(-1,x,Q2)+f(2,x,Q2)+f(3,x,Q2)-f(-4,x,Q2))
      END FUNCTION xF3_nun

      REAL(KIND=dp) FUNCTION F2_nbn(x,Q2)
         REAL(KIND=dp),INTENT(IN) :: x,Q2 
         F2_nbn=2.0_dp*x*(f(1,x,Q2)+f(-2,x,Q2)+f(-3,x,Q2)+f(4,x,Q2))
      END FUNCTION F2_nbn

      REAL(KIND=dp) FUNCTION FL_nbn(x,Q2)
         REAL(KIND=dp),INTENT(IN) :: x,Q2
         FL_nbn=0.0_dp
      END FUNCTION FL_nbn

      REAL(KIND=dp) FUNCTION xF3_nbn(x,Q2)
         REAL(KIND=dp),INTENT(IN) :: x,Q2 
         xF3_nbn=2.0_dp*x*(f(1,x,Q2)-f(-2,x,Q2)-f(-3,x,Q2)+f(4,x,Q2))
      END FUNCTION xF3_nbn
END MODULE structure_functions

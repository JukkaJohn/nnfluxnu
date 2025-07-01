MODULE interpolation

   USE kinds
   
   IMPLICIT NONE

   TYPE interpolation_grid
      REAL(KIND=dp),DIMENSION(:),ALLOCATABLE :: x,fx
      INTEGER(KIND=4) :: BLOCKSIZE=7
      CONTAINS
         PROCEDURE,PASS(self) :: p=>poly
         PROCEDURE,PASS(self) :: f=>interpolated_function
         PROCEDURE,PASS(self) :: sample_poly=>sample_poly
   END TYPE interpolation_grid

   PUBLIC :: interpolation_grid,new_xgrid,logspaced_grid
   PRIVATE :: poly


   TYPE(interpolation_grid)     :: intrpln

   CONTAINS

      SUBROUTINE init_interpolation_grid(xgrid)
         REAL(KIND=dp),DIMENSION(:),INTENT(IN) :: xgrid
         intrpln=new_xgrid(xgrid,xgrid)
      END SUBROUTINE init_interpolation_grid

      SUBROUTINE logspaced_grid(n,lowx,x)
         INTEGER(KIND=4),INTENT(IN) :: n,lowx
         REAL(KIND=dp),DIMENSION(n),INTENT(OUT) :: x
         INTEGER(KIND=4) :: i
         REAL(KIND=dp) :: incexp ! increment of the exponent.
         REAL(KIND=dp) :: rlowx,ri,rn
         IF(lowx.GE.0)THEN
            WRITE(*,"(A)") "Error in logspaced_grid:"
            WRITE(*,"(A)") "Lower power of x is greater than zero."
            STOP
         END IF
         rn=REAL(n,KIND=dp)
         rlowx=REAL(lowx,KIND=dp)
         incexp=rlowx/rn
         DO i=LBOUND(x,1),UBOUND(x,1)
            ri=REAL(i-1,KIND=dp)
            x(i)=DEXP(lowx-ri*incexp)
            write(*,*) i, x(i)
         END DO
         x(n)=1.0_dp ! set to exactly one.
      END SUBROUTINE logspaced_grid

      SUBROUTINE logspaced_subgrids(n1,n2,lowx1,lowx2,x)
         INTEGER(KIND=4),INTENT(IN) :: n1,n2,lowx1,lowx2
         REAL(KIND=dp),DIMENSION(n1+n2),INTENT(OUT) :: x
         INTEGER(KIND=4) :: i
         REAL(KIND=dp) :: inc1,inc2
         REAL(KIND=dp) :: rlowx1,rlowx2,ri,rn1,rn2
         rn1=REAL(n1,KIND=dp)
         rn2=REAL(n2,KIND=dp)
         rlowx1=REAL(lowx1,KIND=dp)
         rlowx2=REAL(lowx2,KIND=dp)
         inc1=(rlowx1-rlowx2)/rn1
         inc2=rlowx2/rn2
         write(*,*) "logspaced"
         write(*,*) rn1, rn2, inc1, inc2
         DO i=1,n1
            ri=REAL(i-1,KIND=dp)
            x(i)=DEXP(lowx1-inc1*ri)
            write(*,*) i, x(i)
         END DO
         DO i=1,n2
            ri=REAL(i-1,KIND=dp)
            x(i+n1)=DEXP(lowx2-inc2*ri)
            write(*,*) i+n1, x(i+n1)
         END DO
      END SUBROUTINE logspaced_subgrids

      TYPE(interpolation_grid) FUNCTION new_xgrid(arr,farr,bs) RESULT(res)
         REAL(KIND=dp),DIMENSION(:),INTENT(IN) :: arr,farr
         INTEGER(KIND=4),OPTIONAL :: bs
         ALLOCATE(res%x,SOURCE=arr)
         ALLOCATE(res%fx,SOURCE=farr)
         IF(PRESENT(bs)) res%BLOCKSIZE=bs
      END FUNCTION new_xgrid
      
      REAL(KIND=dp) FUNCTION interpolated_function(self,x) RESULT(res)
         CLASS(interpolation_grid),INTENT(IN) :: self
         REAL(KIND=dp),INTENT(IN) :: x 
         INTEGER(KIND=4) :: i
         ! Check if x is between the lower and uppers bound of the 
         ! interpolation grid
         IF(x.LT.self%x(LBOUND(self%x,1)).OR. &
            x.GT.self%x(UBOUND(self%x,1)))THEN
            WRITE(*,*) "ERROR in interpolation x is out of bounds:", &
                       LBOUND(self%x,1),x,LBOUND(self%x,1)
            STOP
         END IF
         ! Compute the interpolated value for f(x)
         res=0.0_dp
         DO i=LBOUND(self%x,1),UBOUND(self%x,1)
            res=res+self%fx(i)*self%p(x,i)
         END DO
         RETURN
      END FUNCTION interpolated_function

      ! Lagrange interpolation in log(x), like in EKO.
      REAL(KIND=dp) FUNCTION poly(self,x,j) RESULT(res)
         CLASS(interpolation_grid),INTENT(IN) :: self
         REAL(KIND=dp),INTENT(IN) :: x
         INTEGER(KIND=4),INTENT(IN) :: j
         INTEGER(KIND=4) :: area
         INTEGER(KIND=4) :: l,u,i,n
         area=FINDLOC(self%x,MAXVAL(self%x-x, &
              MASK=self%x-x.LT.-EPSILON(0.0_dp))+x,DIM=1)
         !WRITE(*,*) area,j
         n=SIZE(self%x,1)
         l=self%BLOCKSIZE/2
         u=(self%BLOCKSIZE-1)/2
         DO i=l,1,-1
            IF(area-i.GT.0)THEN
               l=i
               EXIT
            ELSE
               l=l-1
               u=u+1
            END IF
         END DO
         DO i=u,1,-1
            IF(area+i.LE.n)THEN
               u=i
               EXIT
            ELSE
               l=l+1
               u=u-1
            END IF
         END DO
         IF(j.LT.area-l.OR.j.GT.area+u)THEN
            res=0.0_dp
            RETURN
         END IF
         res=1.0_dp
         DO i=area-l,area+u
            IF(i.NE.j)THEN
               !res=res*(x-self%x(i))/(self%x(j)-self%x(i))
               res=res*(LOG(x)-LOG(self%x(i)))/(LOG(self%x(j))-LOG(self%x(i)))
            END IF
         END DO
         RETURN
      END FUNCTION poly

      SUBROUTINE sample_poly(self,ipoly,xmin,xmax)
         CLASS(interpolation_grid),INTENT(IN) :: self
         INTEGER(KIND=4),INTENT(IN) :: ipoly
         REAL(KIND=dp),INTENT(OUT) :: xmin,xmax
         INTEGER(KIND=4) :: i,l,u,n

         l=self%BLOCKSIZE/2
         u=(self%BLOCKSIZE-1)/2
         n=SIZE(self%x,1)

         DO i=l,1,-1
            IF(ipoly-i.GT.0)THEN
               l=i
               EXIT
            ELSE
               l=l-1
               u=u+1
            END IF
         END DO
         DO i=u,1,-1
            IF(ipoly+u.LE.n)THEN
               u=i
               EXIT
            ELSE
               l=l+1
               u=u-1
            END IF
         END DO
         xmin=self%x(ipoly-l)
         xmax=self%x(ipoly+u)
      END SUBROUTINE sample_poly

END MODULE interpolation

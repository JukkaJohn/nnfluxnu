PROGRAM interp_driver
   USE interpolation
   IMPLICIT NONE

   INTEGER, PARAMETER :: n = 20
   REAL(KIND=dp), DIMENSION(n) :: x, fx
   TYPE(interpolation_grid) :: grid
   INTEGER :: i
   REAL(KIND=dp) :: xi, fxi
   INTEGER, PARAMETER :: m = 100
   REAL(KIND=dp) :: xmin, xmax, step

   CALL logspaced_grid(n, -4, x)
   fx = SIN(x)  ! use any function
   grid = new_xgrid(x, x)

   OPEN(10, FILE='interp_output.dat', STATUS='REPLACE')
   xmin = MINVAL(x)
   xmax = MAXVAL(x)
   step = (xmax - xmin) / (m - 1)

   DO i = 0, m-1
      xi = xmin + i * step
      fxi = grid%f(xi)
      WRITE(10,'(F12.6,1X,F12.6)') xi, fxi
   END DO

   CLOSE(10)
END PROGRAM interp_driver

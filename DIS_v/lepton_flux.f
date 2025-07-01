      subroutine sample_x_lepton(random, xmin, xmax, xlep, jac) 
      implicit none
! input
      real *8 random       ! Random number
      real *8 xmin, xmax   ! Lower and upper limit on xlepton (computed in Born_phsp)
!     output
      real *8 xlep         ! Incoming lepton x fraction
      real *8 jac          ! Jacobian from xlep to random
      
!     REPLACE THIS with the importance sampling of your choice
      xlep = xmin + random * (xmax-xmin)
!     jacobian = ∂xlep/∂random
      jac = (xmax-xmin)
      end

      subroutine sample_x_lepton_poly(random,ipoly,xlep,jac,xmin,xmax)
         use interpolation
         implicit none
         real(kind=8),intent(in)  :: random
         integer,intent(in)       :: ipoly
         real(kind=8),intent(out) :: xlep
         real(kind=8),intent(out) :: jac
         real(kind=8),intent(out) :: xmin,xmax
         call intrpln%sample_poly(ipoly,xmin,xmax)
         xlep=xmin+random*(xmax-xmin)
         jac=xmax-xmin
      end 


      subroutine pdf_lepton_beam(ih, mufact2, x, pdf )
      implicit none
      
      include 'pwhg_pdf.h'
      real * 8 lam5
      integer iord,iset,maxsets
      common/cgenericpdf/lam5,iord,iset,maxsets

      integer(kind=4),intent(in) :: ih        !Index of the incoming beam
      real(kind=8),intent(in)    :: mufact2   !factorisation scale (squared)
      real(kind=8),intent(in)    :: x         !x fraction
      real(kind=8),dimension(-pdf_nparton:pdf_nparton),
     1 intent(out)               :: pdf

      integer(kind=4),save       :: nupdf=-1
      logical,save               :: ini=.TRUE.

      interface
         real(kind=8) function powheginput(stringa)
            character(len=*),dimension(*),intent(in) :: stringa
         end function powheginput
         logical(kind=c_bool) function generic_has_id(iset,id) 
     1   bind(C,name="generic_has_id_")
            use iso_c_binding, only: c_int, c_bool
            integer(kind=c_int),intent(in) :: iset,id
         end function generic_has_id
      end interface

      pdf = 1e-8                ! Do not set light quark pdfs to 0 otherwise
                                ! the checklims is broken. This is in any 
                                ! case not used
      if(ini)then
         nupdf=int(powheginput("#nupdf"),kind=4)
         if(nupdf.lt.0)then
            write(*,"(A)") "ERROR: Keyword nupdf missing in input card."
            stop
         end if
         call genericpdfset(nupdf)
         if(.NOT.generic_has_id(iset,ih))then
            write(*,"(A23,I4,A15)") "ERROR: Neutrino flavour",ih, 
     1                            " not available."
            stop
         end if
         ini=.FALSE.
      end if

      call genericpdfset(nupdf)
      call xf_pdgid(iset,ih,x,mufact2,pdf(ih))
      pdf(ih)=pdf(ih)/x

      end

C     Replace the neutrino flux by an interpolation polynomial
      subroutine pdf_lepton_interpolation_beam(ih,mufact2,x,pdf)

      use interpolation
      implicit none

      include 'pwhg_pdf.h'
      real * 8 lam5
      integer iord,iset,maxsets
      common/cgenericpdf/lam5,iord,iset,maxsets

      integer(kind=4),intent(in) :: ih        !Index of the incoming beam
      real(kind=8),intent(in)    :: mufact2   !factorisation scale (squared)
      real(kind=8),intent(in)    :: x         !x fraction
      real(kind=8),dimension(-pdf_nparton:pdf_nparton),
     1 intent(out)               :: pdf

      integer(kind=4),save         :: ipoly=-1
      integer,parameter            :: NARR=50
      real(kind=8),dimension(NARR) :: arr

      logical,save               :: ini=.TRUE.

      integer :: i

      interface
         real(kind=8) function powheginput(stringa)
            character(len=*),dimension(*),intent(in) :: stringa
         end function powheginput
      end interface

      if(ini)then
         ! Set up the interpolation polynomial.
         write(*,"(A)") "Setting up interpolation grid."
         !call logspaced_grid(NARR,-5,arr)
         call logspaced_subgrids(NARR/2,NARR/2,-5,-2,arr)
         if(.not.allocated(intrpln%x)) call init_interpolation_grid(arr)
         ipoly=int(powheginput("#ipoly"),kind=4)
         ini=.FALSE.
      end if

      pdf=1d-8

      pdf(ih)=intrpln%p(x,ipoly)
      end

      subroutine init_pdf_lepton_interpolation_beam()

      use interpolation
      implicit none

      include 'pwhg_pdf.h'
      real * 8 lam5
      integer iord,iset,maxsets
      common/cgenericpdf/lam5,iord,iset,maxsets
      integer(kind=4),save         :: ipoly=-1
      integer,parameter            :: NARR=50
      real(kind=8),dimension(NARR) :: arr

      logical,save               :: ini=.TRUE.

      integer :: i

      interface
         real(kind=8) function powheginput(stringa)
            character(len=*),dimension(*),intent(in) :: stringa
         end function powheginput
      end interface

      if(ini)then
         ! Set up the interpolation polynomial.
         write(*,"(A)") "Setting up interpolation grid."
         !call logspaced_grid(NARR,-5,arr)
         call logspaced_subgrids(NARR/2,NARR/2,-5,-2,arr)
         call init_interpolation_grid(arr)
         ipoly=int(powheginput("#ipoly"),kind=4)
         if(ipoly.lt.0)then
            write(*,*) "ERROR: Keyword alpha missing in input card."
            stop
         end if
         ini=.FALSE.
      end if
      end 

MODULE pineappl

   USE iso_c_binding, ONLY: c_null_ptr,c_ptr,c_size_t
   USE kinds, ONLY: dp,PATHLEN

   IMPLICIT NONE

   TYPE pineappl_grid
      TYPE(c_ptr) :: ptr=c_null_ptr
!      CONTAINS
!         FINAL :: deallocate_grid
   END TYPE pineappl_grid
   TYPE, EXTENDS(pineappl_grid) :: pineappl_fktable
      !TYPE(c_ptr) :: ptr=c_null_ptr
      !CONTAINS
      !   FINAL :: deallocate_fk
   END TYPE pineappl_fktable

   TYPE pineappl_lumi
      TYPE(c_ptr) :: ptr=c_null_ptr
!      CONTAINS
!         FINAL :: deallocate_lumi
   END TYPE pineappl_lumi

   TYPE pineappl_keyval
      TYPE(c_ptr) :: ptr=c_null_ptr
   END TYPE pineappl_keyval

   INTERFACE

      SUBROUTINE pineappl_grid_set_key_value(grid,key,val) &
            BIND(C,NAME="pineappl_grid_set_key_value")
         USE iso_c_binding
         TYPE(c_ptr),VALUE :: grid
         CHARACTER(c_char),DIMENSION(*) :: key,val
      END SUBROUTINE pineappl_grid_set_key_value

      SUBROUTINE pineappl_lumi_add(lumi,combinations,pdg_id_pairs,factors) &
            BIND(C,NAME="pineappl_lumi_add")
         USE iso_c_binding
         TYPE(c_ptr),VALUE :: lumi
         INTEGER(c_size_t),VALUE :: combinations
         INTEGER(c_int32_t),DIMENSION(*) :: pdg_id_pairs
         REAL(c_double),DIMENSION(*) :: factors
      END SUBROUTINE pineappl_lumi_add

      SUBROUTINE pineappl_grid_fill_all(grid,x1,x2,q2,order,observable,weights) &
            BIND(C,NAME="pineappl_grid_fill_all")
         USE iso_c_binding
         TYPE(c_ptr),VALUE :: grid
         REAL(KIND=c_double),VALUE :: x1,x2,q2,observable
         REAL(KIND=c_double),DIMENSION(*) :: weights
         INTEGER(KIND=c_size_t),VALUE :: order
      END SUBROUTINE pineappl_grid_fill_all

      SUBROUTINE pineappl_grid_fill(grid,x1,x2,q2,order,observable,lumi,weight) & 
            BIND(C,NAME="pineappl_grid_fill")
         USE iso_c_binding
         TYPE(c_ptr),VALUE :: grid
         REAL(KIND=c_double),VALUE :: x1,x2,q2,observable,weight
         INTEGER(KIND=c_size_t),VALUE :: order,lumi
      END SUBROUTINE pineappl_grid_fill

      SUBROUTINE pineappl_grid_write(grid,filename) & 
            BIND(C,NAME="pineappl_grid_write")
         USE iso_c_binding
         TYPE(c_ptr),VALUE :: grid
         CHARACTER(c_char),DIMENSION(*) :: filename
      END SUBROUTINE pineappl_grid_write

      SUBROUTINE c_free(ptr) BIND(C,NAME="free")
         USE iso_c_binding, ONLY: c_ptr
         IMPLICIT NONE
         TYPE(c_ptr), VALUE :: ptr
      END SUBROUTINE c_free

      TYPE(c_ptr) FUNCTION pineappl_grid_read(filename) &
            BIND(C,NAME="pineappl_grid_read")
         USE iso_c_binding, ONLY: c_ptr,c_char
         CHARACTER(KIND=c_char) :: filename
      END FUNCTION pineappl_grid_read

      TYPE(c_ptr) FUNCTION pineappl_fktable_read(filename) &
            BIND(C,NAME="pineappl_fktable_read")
         USE iso_c_binding, ONLY: c_ptr,c_char
         CHARACTER(c_char) :: filename
      END FUNCTION pineappl_fktable_read

      SUBROUTINE pineappl_fktable_close(fk) &
            BIND(C,NAME="pineappl_fktable_close")
         USE iso_c_binding, ONLY: c_ptr
         TYPE(c_ptr),VALUE :: fk
      END SUBROUTINE pineappl_fktable_close

      INTEGER(KIND=c_size_t) FUNCTION pineappl_grid_bin_count(grid) &
            BIND(C,NAME="pineappl_grid_bin_count")
         USE iso_c_binding, ONLY: c_size_t,c_ptr
         TYPE(c_ptr),VALUE :: grid
      END FUNCTION pineappl_grid_bin_count

      INTEGER(KIND=c_size_t) FUNCTION pineappl_grid_bin_dimensions(grid) &
            BIND(C,NAME="pineappl_grid_bin_dimensions")
         USE iso_c_binding, ONLY: c_size_t,c_ptr
         TYPE(c_ptr),VALUE :: grid
      END FUNCTION pineappl_grid_bin_dimensions

!      REAL(KIND=8) FUNCTION get_xgrid_val(fk,i) &
!            BIND(C,NAME="get_xgrid_val")
!         USE iso_c_binding
!         TYPE(c_ptr),VALUE :: fk
!         INTEGER(c_int32_t),VALUE :: i
!      END FUNCTION get_xgrid_val

      ! Passing an integer around between fortran, c and rust is easier.
      INTEGER(KIND=c_int32_t) FUNCTION pineappl_get_has_pdf1(fk) &
            BIND(C,NAME="get_has_pdf1")
         USE iso_c_binding
         TYPE(c_ptr),VALUE :: fk
      END FUNCTION pineappl_get_has_pdf1

      INTEGER(KIND=c_int32_t) FUNCTION pineappl_get_has_pdf2(fk) &
            BIND(C,NAME="get_has_pdf2")
         USE iso_c_binding
         TYPE(c_ptr),VALUE :: fk
      END FUNCTION pineappl_get_has_pdf2

      INTEGER(KIND=c_int32_t) FUNCTION get_xgrid_length(fk) &
            BIND(C,NAME="get_xgrid_length")
         USE iso_c_binding
         TYPE(c_ptr),VALUE :: fk
      END FUNCTION get_xgrid_length

      INTEGER(KIND=c_int32_t) FUNCTION get_lumi_length(fk) &
            BIND(C,NAME="get_lumi_length")
         USE iso_c_binding
         TYPE(c_ptr),VALUE :: fk
      END FUNCTION get_lumi_length

      REAL(KIND=c_double) FUNCTION pineappl_get_muf2(fk) &
            BIND(C,NAME="get_muf2")
         USE iso_c_binding
         TYPE(c_ptr),VALUE :: fk
      END FUNCTION pineappl_get_muf2

      REAL(KIND=c_double) FUNCTION pineappl_get_bin_norm(fk,i) &
            BIND(C,NAME="get_bin_norm")
         USE iso_c_binding
         TYPE(c_ptr),VALUE :: fk
         INTEGER(c_int32_t),VALUE :: i
      END FUNCTION pineappl_get_bin_norm

      SUBROUTINE pineappl_grid_bin_limits_left(fk,d,arr) &
            BIND(C,NAME="pineappl_grid_bin_limits_left")
         USE iso_c_binding
         TYPE(c_ptr),VALUE :: fk
         INTEGER(c_size_t),VALUE :: d
         !REAL(KIND=c_double),DIMENSION(*) :: arr
         TYPE(c_ptr),VALUE :: arr
      END SUBROUTINE pineappl_grid_bin_limits_left

      SUBROUTINE pineappl_grid_bin_limits_right(fk,d,arr) &
            BIND(C,NAME=" pineappl_grid_bin_limits_right")
         USE iso_c_binding
         TYPE(c_ptr),VALUE :: fk
         INTEGER(c_size_t),VALUE :: d
         !REAL(KIND=c_double),DIMENSION(*) :: arr
         TYPE(c_ptr),VALUE :: arr
      END SUBROUTINE pineappl_grid_bin_limits_right

      INTEGER(KIND=c_int32_t) FUNCTION get_lumi_pid0(fk,i) &
            BIND(C,NAME="get_lumi_pid0")
         USE iso_c_binding
         TYPE(c_ptr),VALUE :: fk
         INTEGER(c_size_t),VALUE :: i
      END FUNCTION get_lumi_pid0
      INTEGER(KIND=c_int32_t) FUNCTION get_lumi_pid1(fk,i) &
            BIND(C,NAME="get_lumi_pid1")
         USE iso_c_binding
         TYPE(c_ptr),VALUE :: fk
         INTEGER(c_size_t),VALUE :: i
      END FUNCTION get_lumi_pid1

      SUBROUTINE pineappl_xgrid(fk,xgrid) &
         BIND(C,NAME="pineappl_xgrid")
         USE iso_c_binding
         TYPE(c_ptr),VALUE :: fk
         !REAL(KIND=c_double),DIMENSION(*) :: xgrid
         TYPE(c_ptr),VALUE :: xgrid
      END SUBROUTINE pineappl_xgrid

!      SUBROUTINE pineappl_get_table(fk,table) &
!         BIND(C,NAME="get_table")
!         USE iso_c_binding
!         TYPE(c_ptr),INTENT(IN),VALUE :: fk
!         REAL(KIND=c_double),DIMENSION(*),INTENT(INOUT) :: table
!      END SUBROUTINE pineappl_get_table
      SUBROUTINE pineappl_get_table(fk,table) &
         BIND(C,NAME="get_table")
         USE iso_c_binding
         TYPE(c_ptr),INTENT(IN),VALUE :: fk
         TYPE(c_ptr),INTENT(IN),VALUE :: table
      END SUBROUTINE pineappl_get_table

      TYPE(c_ptr) FUNCTION pineappl_grid_new(lumi,orders,order_params,&
                                             bins,bin_limits,key_vals)&
                                             BIND(C,NAME="pineappl_grid_new")
         USE iso_c_binding
         TYPE(c_ptr),VALUE :: lumi,key_vals
         INTEGER(c_size_t),VALUE :: orders,bins
         INTEGER(c_int32_t),DIMENSION(*) :: order_params
         REAL(c_double),DIMENSION(*) :: bin_limits
      END FUNCTION pineappl_grid_new

      TYPE(c_ptr) FUNCTION pineappl_keyval_new() &
            BIND(C,NAME="pineappl_keyval_new")
         USE iso_c_binding
      END FUNCTION pineappl_keyval_new

      TYPE(c_ptr) FUNCTION pineappl_lumi_new() &
            BIND(C,NAME="pineappl_lumi_new")
         USE iso_c_binding
      END FUNCTION pineappl_lumi_new

   END INTERFACE


   CONTAINS

      SUBROUTINE grid_fill_all(grid,x1,x2,q2,order,observable,weights)
         TYPE(pineappl_grid),INTENT(IN) :: grid
         REAL(KIND=dp),INTENT(IN) :: x1,x2,q2,observable
         INTEGER(KIND=4),INTENT(IN) :: order
         REAL(KIND=dp),DIMENSION(*) :: weights
         CALL pineappl_grid_fill_all(grid%ptr,x1,x2,q2, &
            INT(order,KIND=c_size_t),observable,weights)
      END SUBROUTINE grid_fill_all

      SUBROUTINE grid_fill(grid,x1,x2,q2,order,observable,lumi,weight)
         TYPE(pineappl_grid),INTENT(IN) :: grid
         REAL(KIND=dp),INTENT(IN) :: x1,x2,q2,observable,weight
         INTEGER(KIND=4),INTENT(IN) :: order,lumi
         CALL pineappl_grid_fill(grid%ptr,x1,x2,q2, &
            INT(order,KIND=c_size_t),observable,INT(lumi,KIND=c_size_t),weight)
      END SUBROUTINE grid_fill

      SUBROUTINE grid_write(grid,filename)
         TYPE(pineappl_grid),INTENT(IN) :: grid
         CHARACTER(LEN=*),INTENT(IN) :: filename
         CALL pineappl_grid_write(grid%ptr,filename)
      END SUBROUTINE grid_write
      
      SUBROUTINE lumi_add(lumi,combinations,pdg_id_pairs,factors)
         TYPE(pineappl_lumi),INTENT(IN) :: lumi
         INTEGER(KIND=4),INTENT(IN) :: combinations
         INTEGER(KIND=4),DIMENSION(2*combinations),INTENT(IN) :: pdg_id_pairs
         REAL(KIND=dp),DIMENSION(combinations),INTENT(IN) :: factors

         CALL pineappl_lumi_add(lumi%ptr,INT(combinations,KIND=c_size_t), &
            pdg_id_pairs, factors)
      END SUBROUTINE lumi_add

      TYPE(pineappl_lumi) FUNCTION lumi_new()
         lumi_new=pineappl_lumi(pineappl_lumi_new())
      END FUNCTION lumi_new

      TYPE(pineappl_keyval) FUNCTION keyval_new()
         keyval_new=pineappl_keyval(pineappl_keyval_new())
      END FUNCTION keyval_new

      SUBROUTINE pineappl_keyval_set_string(keyval,key,val) &
            BIND(C,NAME="pineappl_keyval_set_string")
         USE iso_c_binding
         TYPE(c_ptr),VALUE :: keyval
         CHARACTER(c_char),DIMENSION(*) :: key,val
      END SUBROUTINE pineappl_keyval_set_string

      SUBROUTINE pineappl_keyval_set_int(keyval,key,val) &
            BIND(C,NAME="pineappl_keyval_set_int")
         USE iso_c_binding
         TYPE(c_ptr),VALUE :: keyval
         CHARACTER(c_char),DIMENSION(*) :: key
         INTEGER(KIND=c_int32_t),VALUE :: val
      END SUBROUTINE pineappl_keyval_set_int
      SUBROUTINE pineappl_keyval_set_double(keyval,key,val) &
            BIND(C,NAME="pineappl_keyval_set_double")
         USE iso_c_binding
         TYPE(c_ptr),VALUE :: keyval
         CHARACTER(c_char),DIMENSION(*) :: key
         REAL(KIND=c_double),VALUE :: val
      END SUBROUTINE pineappl_keyval_set_double

      TYPE(pineappl_grid) FUNCTION grid_new(lumi,orders,order_params,bins,bin_limits,key_vals)
         USE iso_c_binding
         TYPE(pineappl_lumi),INTENT(IN) :: lumi
         INTEGER(KIND=4),INTENT(IN) :: orders,bins
         INTEGER(KIND=4),DIMENSION(4*orders),INTENT(IN) :: order_params
         REAL(KIND=dp),DIMENSION(bins+1),INTENT(IN) :: bin_limits
         TYPE(pineappl_keyval),INTENT(IN) :: key_vals
         grid_new=pineappl_grid(pineappl_grid_new(lumi%ptr, &
            INT(orders,KIND=c_size_t),order_params,INT(bins,KIND=c_size_t), &
            bin_limits, key_vals%ptr))
      END FUNCTION grid_new

      IMPURE ELEMENTAL SUBROUTINE deallocate_grid(self)
         USE iso_c_binding
         TYPE(pineappl_grid),INTENT(INOUT) :: self
         CALL c_free(self%ptr)
         self%ptr=c_null_ptr
      END SUBROUTINE deallocate_grid
      IMPURE ELEMENTAL SUBROUTINE deallocate_fk(self)
         USE iso_c_binding
         TYPE(pineappl_fktable),INTENT(INOUT) :: self
         IF(.NOT.C_ASSOCIATED(self%ptr)) THEN
            CALL pineappl_fktable_close(self%ptr)
         END IF
      END SUBROUTINE deallocate_fk
      IMPURE ELEMENTAL SUBROUTINE deallocate_lumi(self)
         USE iso_c_binding
         TYPE(pineappl_lumi),INTENT(INOUT) :: self
         CALL c_free(self%ptr)
         self%ptr=c_null_ptr
      END SUBROUTINE deallocate_lumi

      TYPE(pineappl_grid) FUNCTION read_grid(filename)
         USE iso_c_binding
         CHARACTER(LEN=*) :: filename
         read_grid = pineappl_grid(pineappl_grid_read(filename))
      END FUNCTION read_grid

      TYPE(pineappl_fktable) FUNCTION read_fktable(filename)
         USE iso_c_binding
         CHARACTER(LEN=PATHLEN) :: filename
         INTEGER(KIND=4) :: i
         LOGICAL :: exists
         i = LEN_TRIM(filename)+1
         INQUIRE(FILE=ADJUSTL(TRIM(filename)),EXIST=exists)
         IF(.NOT.exists) THEN 
            WRITE(*,*) "FkTable not found:",ADJUSTL(TRIM(filename))
            STOP
         END IF
         filename(i:i) = c_null_char
         read_fktable=pineappl_fktable(ptr=pineappl_fktable_read(filename))
      END FUNCTION read_fktable

      INTEGER(KIND=4) FUNCTION get_grid_bin_count(grid)
         USE iso_c_binding
         CLASS(pineappl_grid),INTENT(IN) :: grid
         get_grid_bin_count=INT(pineappl_grid_bin_count(grid%ptr),KIND=4)
      END FUNCTION get_grid_bin_count

      INTEGER(KIND=4) FUNCTION get_grid_bin_dimensions(grid)
         USE iso_c_binding
         CLASS(pineappl_grid),INTENT(IN) :: grid
         get_grid_bin_dimensions=INT(pineappl_grid_bin_dimensions(grid%ptr),KIND=4)
      END FUNCTION get_grid_bin_dimensions


      SUBROUTINE get_xgrid(fk,xgrid)
         USE iso_c_binding
         TYPE(pineappl_fktable),INTENT(IN) :: fk
         REAL(KIND=dp),DIMENSION(:),INTENT(INOUT),TARGET :: xgrid
         TYPE(c_ptr) :: cptr
         cptr=C_LOC(xgrid(1))
         !CALL pineappl_xgrid(fk%ptr,xgrid)
         CALL pineappl_xgrid(fk%ptr,cptr)
      END SUBROUTINE get_xgrid

!      REAL(KIND=dp) FUNCTION get_xgrid_value(fk,i)
!         USE iso_c_binding
!         TYPE(pineappl_fktable),INTENT(IN) :: fk
!         INTEGER(KIND=4),INTENT(IN) :: i
!         get_xgrid_value = get_xgrid_val(fk%ptr,INT(i,KIND=c_int32_t))
!      END FUNCTION get_xgrid_value

      INTEGER(KIND=4) FUNCTION get_xgrid_len(fk)
         USE iso_c_binding
         TYPE(pineappl_fktable),INTENT(IN) :: fk
         get_xgrid_len = INT(get_xgrid_length(fk%ptr),KIND=4)
      END FUNCTION get_xgrid_len

      INTEGER(KIND=4) FUNCTION get_lumi_len(fk)
         USE iso_c_binding
         TYPE(pineappl_fktable),INTENT(IN) :: fk
         get_lumi_len = INT(get_lumi_length(fk%ptr),KIND=4)
      END FUNCTION get_lumi_len

      REAL(KIND=dp) FUNCTION get_muf2(fk)
         USE iso_c_binding
         TYPE(pineappl_fktable),INTENT(IN) :: fk
         get_muf2 = REAL(pineappl_get_muf2(fk%ptr),KIND=dp)
      END FUNCTION get_muf2

      REAL(KIND=dp) FUNCTION get_bin_norm(fk,i)
         USE iso_c_binding
         TYPE(pineappl_fktable),INTENT(IN) :: fk
         INTEGER(KIND=4) :: i
         get_bin_norm = REAL(pineappl_get_bin_norm(fk%ptr,i-1),KIND=dp)
      END FUNCTION get_bin_norm

      SUBROUTINE get_left_lims(fk,d,arr)
         USE iso_c_binding
         TYPE(pineappl_fktable),INTENT(IN) :: fk
         INTEGER(KIND=4) :: d ! possible dimensions are 0 or 1 
         TYPE(c_ptr) :: ptr
         !REAL(KIND=dp),DIMENSION(*),INTENT(INOUT),TARGET :: arr
         REAL(KIND=dp),DIMENSION(:),INTENT(INOUT),TARGET :: arr
         ptr=C_LOC(arr(1))
         !CALL pineappl_grid_bin_limits_left(fk%ptr,INT(d,KIND=c_size_t),arr)
         CALL pineappl_grid_bin_limits_left(fk%ptr,INT(d,KIND=c_size_t),ptr)
      END SUBROUTINE get_left_lims

      SUBROUTINE get_right_lims(fk,d,arr)
         USE iso_c_binding
         TYPE(pineappl_fktable),INTENT(IN) :: fk
         TYPE(c_ptr) :: ptr
         INTEGER(KIND=4) :: d ! possible dimensions are 0 or 1 
         !REAL(KIND=dp),DIMENSION(*),INTENT(INOUT),TARGET :: arr
         REAL(KIND=dp),DIMENSION(:),INTENT(INOUT),TARGET :: arr
         ptr=C_LOC(arr(1))
         !CALL pineappl_grid_bin_limits_right(fk%ptr,INT(d,KIND=c_size_t),arr)
         CALL pineappl_grid_bin_limits_right(fk%ptr,INT(d,KIND=c_size_t),ptr)
      END SUBROUTINE get_right_lims

      INTEGER(KIND=4) FUNCTION get_pid(fk,i)
         USE iso_c_binding
         TYPE(pineappl_fktable),INTENT(IN) :: fk
         INTEGER(KIND=8) :: i
         get_pid = INT(get_lumi_pid0(fk%ptr,INT(i-1,KIND=c_size_t)),KIND=4)
      END FUNCTION get_pid

      SUBROUTINE get_pids(fk,i,pids)
         USE iso_c_binding
         TYPE(pineappl_fktable),INTENT(IN) :: fk
         INTEGER(KIND=4),DIMENSION(2),INTENT(INOUT) :: pids
         INTEGER(KIND=8) :: i
         pids(1) = INT(get_lumi_pid0(fk%ptr,INT(i-1,KIND=c_size_t)),KIND=4)
         pids(2) = INT(get_lumi_pid1(fk%ptr,INT(i-1,KIND=c_size_t)),KIND=4)
      END SUBROUTINE get_pids

      SUBROUTINE get_table(fk,n,table) 
         USE iso_c_binding
         TYPE(pineappl_fktable),INTENT(IN) :: fk
         INTEGER(KIND=8),INTENT(IN) :: n
         REAL(KIND=dp),DIMENSION(:,:,:,:),INTENT(INOUT) :: table
         REAL(KIND=dp),DIMENSION(:),ALLOCATABLE,TARGET :: table1d
         TYPE(c_ptr) :: cptr
         INTEGER(KIND=4) :: ierr
         IF(ALLOCATED(table1d)) DEALLOCATE(table1d)
         ALLOCATE(table1d(n),STAT=ierr)
         IF(ierr.NE.0)THEN
            WRITE(*,*) "Error: Could not allocate array in get_table:",ierr
            STOP
         END IF
         cptr=C_LOC(table1d(1))
         !CALL pineappl_get_table(fk%ptr,table1d)
         table1d(1)=9876.0_dp
         CALL pineappl_get_table(fk%ptr,cptr)
         table=RESHAPE(table1d,SHAPE(table))
         IF(ALLOCATED(table1d)) DEALLOCATE(table1d)
      END SUBROUTINE get_table

      LOGICAL FUNCTION get_has_pdf1(fk)
         USE iso_c_binding
         TYPE(pineappl_fktable),INTENT(IN) :: fk
         INTEGER(KIND=c_int32_t) :: ret
         ret=pineappl_get_has_pdf1(fk%ptr)
         IF(ret.EQ.1) THEN
            get_has_pdf1=.TRUE.
         ELSE IF(ret.EQ.0) THEN
            get_has_pdf1=.FALSE.
         ELSE
            WRITE(*,*) "Could not determine has_pdf1 form FkTable."
            STOP
         END IF
      END FUNCTION get_has_pdf1

      LOGICAL FUNCTION get_has_pdf2(fk)
         USE iso_c_binding
         TYPE(pineappl_fktable),INTENT(IN) :: fk
         INTEGER(KIND=c_int32_t) :: ret
         ret=pineappl_get_has_pdf2(fk%ptr)
         IF(ret.EQ.1) THEN
            get_has_pdf2=.TRUE.
         ELSE IF(ret.EQ.0) THEN
            get_has_pdf2=.FALSE.
         ELSE
            WRITE(*,*) "Could not determine has_pdf2 form FkTable."
            STOP
         END IF
      END FUNCTION get_has_pdf2

      SUBROUTINE keyval_set_string(keyval,key,val)
         USE iso_c_binding, ONLY: c_null_char
         TYPE(pineappl_keyval),INTENT(IN) :: keyval
         CHARACTER(LEN=*),INTENT(IN) :: key,val
         CALL pineappl_keyval_set_string(keyval%ptr,key//c_null_char,val//c_null_char)
      END SUBROUTINE keyval_set_string

      SUBROUTINE keyval_set_int(keyval,key,val)
         USE iso_c_binding, ONLY: c_null_char
         TYPE(pineappl_keyval),INTENT(IN) :: keyval
         CHARACTER(LEN=*),INTENT(IN) :: key
         INTEGER(KIND=4),INTENT(IN) :: val
         CALL pineappl_keyval_set_int(keyval%ptr,key//c_null_char,val)
      END SUBROUTINE keyval_set_int
      SUBROUTINE keyval_set_double(keyval,key,val)
         USE iso_c_binding, ONLY: c_null_char
         TYPE(pineappl_keyval),INTENT(IN) :: keyval
         CHARACTER(LEN=*),INTENT(IN) :: key
         REAL(KIND=dp),INTENT(IN) :: val
         CALL pineappl_keyval_set_double(keyval%ptr,key//c_null_char,val)
      END SUBROUTINE keyval_set_double

      SUBROUTINE grid_set_key_value(grid,key,val) 
         USE iso_c_binding
         TYPE(pineappl_grid),INTENT(IN) :: grid
         CHARACTER(LEN=*),INTENT(IN) :: key,val
         CALL pineappl_grid_set_key_value(grid%ptr,key//c_null_char,val//c_null_char)
      END SUBROUTINE grid_set_key_value


END MODULE pineappl

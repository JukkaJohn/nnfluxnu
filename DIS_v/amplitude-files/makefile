FC = gfortran
#FC = ifort
POWHEGPATH=/Users/krack/POWHEG-BOX-RES
ifeq ("$(shell whoami)", "pkrack")
POWHEGPATH=/data/theorie/pkrack/POWHEG-BOX-RES
endif
ifeq ("$(shell whoami)", "krack")
POWHEGPATH=/Users/krack/POWHEG-BOX-RES
endif

INCLUDE1=$(PWD)
INCLUDE2=$(PWD)/..
INCLUDE3=$(POWHEGPATH)/include

ifeq ("$(FC)","gfortran")	
FFLAGS = -g -fno-automatic -ffixed-line-length-none -O2 -I$(INCLUDE1) -I$(INCLUDE2) -I$(INCLUDE3) 	
endif
ifeq ("$(FC)","ifort")	
FFLAGS = -save -extend_source -O3 -I$(INCLUDE1) -I$(INCLUDE2)  -I$(INCLUDE3) 
endif

# ----------------------------------------------------------------------------

# object files for qq->qqHH

OBJECTS = ./qq_ee.o \
          ./qqj_ee.o \
          ./help_hh.o \
	  ./braket_vbf.o  

TARGETS = libamplitude.a


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# rules for vbfnlo_lib:
#
 libamplitudes.a:  $(OBJECTS)
	$(AR) r libamplitude.a $(OBJECTS) 

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# for g77 or gfortran: 
.f.o:
	$(FC) -c -o $*.o  $(FFLAGS) $*.f

# ----------------------------------------------------------------------------

clean:
	rm -f *.o core *~ *.x libamplitude.a




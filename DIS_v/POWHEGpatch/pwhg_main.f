      program pwhg_main
      implicit none
      integer iwhichseed,parallelstage,xgriditeration
      include 'LesHouches.h'
      include 'nlegborn.h'
      include 'pwhg_flst.h'
      include 'pwhg_rad.h'
      include 'pwhg_st.h'
      include 'pwhg_kn.h'
      include 'pwhg_rnd.h'
      include 'pwhg_flg.h'
      include 'pwhg_par.h'
      include 'pwhg_weights.h'
      include 'pwhg_rwl.h'
      integer j,iun,iunin,iunout,nev,maxev
      character * 10 statlhe
      common/cnev/nev
      real * 8 weight,tmp
      real * 8 powheginput
      character * 20 pwgprefix
      integer lprefix
      common/cpwgprefix/pwgprefix,lprefix
      integer ios,iarg
      character * 6 WHCPRG
      character * 100 filename
      common/cWHCPRG/WHCPRG
      integer iseed,n1,n2
      logical testplots
      integer num_stored_evt,count_stored_evt,k_stored_evt,
     1     num_weights,num_old_weights,count_weights
      real * 8, allocatable :: weights(:,:)
      logical exist
      character * 100 arg
      character * 4 cpoly
      integer ipoly
c     any number of symbol=value on the command line
c     is interpreted as overriding the powheg.input value.
c     Furthermore, the special keyword iwhichseed=<integer>
c     assigns the seed number of parallel runs, preventing
c     the program to inquire for it later.
      iwhichseed = -1
      iarg=1
      call get_command_argument(iarg,arg)
      do while(arg /= '')
         j=index(arg,'=')
         if(j>0) then
            read(arg(j+1:),fmt=*,iostat=ios) tmp
         else
            ios = -1
         endif
         if(ios == 0) then
            select case (arg(1:j-1))
            case ('iwhichseed')
               iwhichseed = nint(tmp)
            case default
               call powheginputoverride(arg(1:j-1),tmp)
            end select
         else
            write(*,*) 'pwhg_main: error in command line argument ',iarg
            call exit(-1)
         endif
         iarg=iarg+1
         call get_command_argument(iarg,arg)
      enddo

      flg_detailedNLO = powheginput("#detailedNLO") == 1
      
c Print out svn information, if any
      iun = 6

      write(*,*)
      write(*,*) "****************************************************************************"
      write(*,*) "****************************************************************************"
      write(*,*) "****                                                                    ****"
      write(*,*) "****          Thank you for using the POWHEG-BOX-RES program!           ****"
      write(*,*) "****                                                                    ****"
      write(*,*) "**** The POWHEG-BOX-RES framework has been developed in                 ****"
      write(*,*) "****                                                                    ****"
      write(*,*) "****           Tomas Jezo and Paolo Nason, JHEP 1512 (2015) 065         ****"
      write(*,*) "****                                                                    ****"
      write(*,*) "**** and is an extension of the POWHEG-BOX-V2 and POWHEG-BOX frameworks ****"
      write(*,*) "****                                                                    ****"
      write(*,*) "****      Simone Alioli, Paolo Nason, Carlo Oleari, Emanuele Re,        ****"
      write(*,*) "****                      JHEP 1006 (2010) 043                          ****"
      write(*,*) "****                                                                    ****"
      write(*,*) "**** based on the POWHEG method                                         ****"
      write(*,*) "****                                                                    ****"
      write(*,*) "****            Stefano Frixione, Paolo Nason, Carlo Oleari,            ****"
      write(*,*) "****                       JHEP 0711 (2007) 070                         ****"
      write(*,*) "****                                                                    ****"
      write(*,*) "**** first introduced in                                                ****"
      write(*,*) "****                                                                    ****"
      write(*,*) "****                Paolo Nason, JHEP 0411 (2004) 040.                  ****"
      write(*,*) "****                                                                    ****"
      include 'ProcessCitation.f'
      write(*,*) "****************************************************************************"
      write(*,*) "****************************************************************************"
      write(*,*)
      
      include 'svn.version'

      if (powheginput('#compress_lhe').eq.1d0) then
         flg_compress_lhe=.true.
      else
         flg_compress_lhe=.false.
      endif

      if (powheginput('#compress_upb').eq.1d0) then
         flg_compress_upb=.true.
      else
         flg_compress_upb=.false.
      endif

c The following can only be changed by a user process for
c testing purposes. Forces a tiny value of st_alpha in setstrongcoupl.f
      flg_tiny_alphas = .false.

      if (powheginput('#testplots').eq.1d0) then
         flg_nlotest=.true.
      else
         flg_nlotest=.false.
      endif
      nev=powheginput('numevts')

c whether to save btilde calls to set up upper bounding envelope
      if(powheginput('#storemintupb').eq.1d0) then
         flg_storemintupb = .true.
      else
         flg_storemintupb = .false.
      endif
c whether to save btilde calls to set up upper bounding envelope
      if(powheginput('#fastbtlbound').eq.1d0) then
         flg_fastbtlbound = .true.
      else
         flg_fastbtlbound = .false.
      endif

c     If set to false, the program has its default behaviour.
c     If set to true, it affects the routines that establish whether a pair
c     of partons can be colour correlated: colcorr() in sigcollsoft.f.
      flg_zerowidth = powheginput("#zerowidth") == 1
c     The following variable should be set to one if the user is perfoming
c     manually its own separation of resonance histories 
      flg_user_reshists_sep = powheginput("#user_reshists_sep") == 1

      par_mintupb_ratlim = powheginput("#mintupbratlim")
      if(par_mintupb_ratlim < 0) par_mintupb_ratlim = 1d50
     
      call newunit(iun)

c The following allows to perform multiple runs with
c different random seeds in the same directory.
c If manyseeds is set to 1, the program asks for an integer j;
c The file 'pwgprefix'seeds.dat at line j is read, and the
c integer at line j is used to initialize the random
c sequence for the generation of the event.
c The event file is called 'pwgprefix'events-'j'.lhe
      if(powheginput("#manyseeds").eq.1) then

         par_maxseeds=powheginput("#maxseeds")
         if(par_maxseeds < 0) then
            par_maxseeds = 9999
         endif

         open(unit=iun,status='old',iostat=ios,
     1        file=pwgprefix(1:lprefix)//'seeds.dat')
          if(ios.ne.0) then
             write(*,*) 'option manyseeds required but '
             write(*,*) 'file ',pwgprefix(1:lprefix)/
     $            /'seeds.dat not found'
            call exit(-1)
         endif 
         do j=1,1000000
            read(iun,*,iostat=ios)  rnd_initialseed
            if(ios.ne.0) goto 10
         enddo
 10      continue
         rnd_numseeds=j-1
         if(iwhichseed > 0) then
            rnd_iwhichseed = iwhichseed
         else
            write(*,*) 'enter which seed'
            read(*,*) rnd_iwhichseed
         endif
         if(rnd_iwhichseed.gt.rnd_numseeds) then
            write(*,*) ' no more than ',rnd_numseeds, ' seeds in ',
     1           pwgprefix(1:lprefix)//'seeds.dat'
            call exit(-1)
         endif
         if(rnd_iwhichseed.gt.par_maxseeds) then
            write(*,*)
     1           ' maximum seed value exceeded ',
     2           rnd_iwhichseed, '>', par_maxseeds
            write(*,*) ' Add to the powheg.input file a line like'
            write(*,*) ' maxseeds <maximum seed you need>'
            call exit(-1)
         endif
         rewind(iun)
         do j=1,rnd_iwhichseed
c Commented line to be used instead, for testing that manyseed runs
c yield the same results as single seed runs, provided the total number
c of calls is the same.
c     read(iun,*) rnd_initialseed,rnd_i1,rnd_i2
            read(iun,*) rnd_initialseed
            rnd_i1=0
            rnd_i2=0
         enddo
         close(iun)
         call seed2string(rnd_iwhichseed,rnd_cwhichseed)
      else
         rnd_cwhichseed='none'
      endif
c If multiple weights may be used in the analysis, set them
c initially to 0 (no multiple weights). These are normally used
c only when reading lh files containing multiple weights.
      weights_num=0
      flg_rwl_add = nint(powheginput("#rwl_add")) == 1
      if(flg_rwl_add) flg_rwl = .true.
      rwl_format_rwgt = nint(powheginput("#rwl_format_rwgt")) == 1
c
      if (flg_nlotest) WHCPRG='NLO   '
      call pwhginit
      if(nev.gt.0) then
         if(powheginput("#ipoly").gt.0)then
            ipoly=powheginput("#ipoly")
            WRITE(cpoly,"(I4.4)") ipoly
         end if
         if(powheginput("#clobberlhe").eq.1) then
            statlhe = 'unknown'
         else
            statlhe = 'new'
         endif
         if(flg_newweight.or.flg_rwl_add) then
            if (flg_nlotest) then 
               write(*,*) '-------> Warning: flg_nlotest has been reset to
     1 false since we are doing reweighting' 
               flg_nlotest = .false. 
            endif
            continue
         else
            if(rnd_cwhichseed.ne.'none') then
               filename=pwgprefix(1:lprefix)
     1              //'events-'//trim(rnd_cwhichseed)//'.lhe'
            else if(ipoly.gt.0)then
               filename=pwgprefix(1:lprefix)//"events-"//TRIM(cpoly)
     1                                      //".lhe"
            else
               filename=pwgprefix(1:lprefix)//'events.lhe'
            endif
            if(powheginput("#clobberlhe").ne.1) then
               inquire(file=filename,exist=exist)
               if(exist) then
                  write(*,*) 'pwhg_main: error, file ',trim(filename),
     1                 ' exists! will not overwrite, exiting ...'
                  call exit(-1)
               endif
            endif
         endif
      else
         write(*,*) ' No events requested'
         goto 999
      endif

c Input the string variables for the standard xml reweight format
c      call getreweightinput  ! obsolete

      if (flg_nlotest) then
         call init_hist 
c     let the analysis subroutine know that it is run by this program
         WHCPRG='LHE   '
      endif
c if we are using manyseeds, and iseed is given, it means that we want
c to examine that event in particular
      if(rnd_cwhichseed.ne.'none') then
         iseed=powheginput('#iseed')
         n1=powheginput('#rand1')
         n2=powheginput('#rand2')
         if(iseed.ge.0.and.n1.ge.0.and.n2.ge.0)
     1        call setrandom(iseed,n1,n2)
      endif
      call resetcnt
     1       ('upper bound failure in inclusive cross section')
      call resetcnt
     1       ('vetoed calls in inclusive cross section')
      call resetcnt(
     1 'upper bound failures in generation of radiation')
      call resetcnt('vetoed radiation')
      write(*,*)
      write(*,*)' POWHEG: generating events'
      flg_fullrwgt = powheginput("#fullrwgt") .eq. 1         
      if(flg_rwl_add) then
         call opencountunit(maxev,iunin)
         call openoutputrw(iunout)
         if(flg_fullrwgt) then
c the following reads the pdf used in the input .lhe file;
c this is needed for fullrwgt.
            call readpowheginputinfo(iunin)
         endif
c     The following copies the input header to the output header. If a <initrwgt>
c     section is found, it is parsed and stored into the rwl common block.
c     Their number is returned in the variable num_old_weights,
c     After this the xml information with the new weights to be added is parsed
c     and stored in the rwl common block. Their number is stored in the variable
c     num_new_weights. The full weight information is written in an xml section
c     <initrwgt>  ... </initrwgt> in the output file header, in increasing group
c     number (group 0 are the weights not belonging to a group) and increasing order.
c     Weights will be written in the same order.
         call rwl_copyheader(iunin,iunout,num_old_weights,num_weights)
         if(maxev.ne.nev) then
            write(*,*) ' Warning: powheg.input says ',nev,' events'
            write(*,*) ' the file contains ', maxev, ' events'
            write(*,*) ' Doing ',maxev,' events'
            nev = maxev 
         endif
      elseif(flg_newweight) then
         call opencountunit(maxev,iunin)
         if(flg_fullrwgt) then
c the following reads the pdf used in the input .lhe file;
c this is needed for fullrwgt.
            call readpowheginputinfo(iunin)
         endif
         call openoutputrw(iunout)
         if(maxev.ne.nev) then
            write(*,*) ' Warning: powheg.input says ',nev,' events'
            write(*,*) ' the file contains ', maxev, ' events'
            write(*,*) ' Doing ',maxev,' events'
            nev = maxev 
         endif
      else
         call rwl_setup(num_weights)
         num_old_weights = 0
         call pwhg_io_open_write(trim(filename),iunout,
     1        flg_compress_lhe,ios)
         call lhefwritehdr(iunout)
      endif
c allocate a weight array of the appropriate size
      call rwl_allocate_weights
      
      flg_noevents = powheginput("#noevents") .eq. 1
      
      testplots = .false.
      if(flg_noevents) then
         testplots = .true.
         write(*,*) 
     1' Since noevents is specified, testplots will be produced'
         write(*,*) ' irrespective of the testplot flag setting.'
      endif
      if(flg_rwl) then
         num_stored_evt = powheginput("#rwl_group_events")
         if(num_stored_evt < 0) num_stored_evt = 1000
         count_stored_evt = 0
         allocate(weights(num_weights,num_stored_evt))
      else
         num_stored_evt = 0
         num_weights = 0
         count_stored_evt = 0
         num_old_weights = 0
         allocate(weights(num_weights,num_stored_evt))
      endif
c     Read weight information
c      call read_weights_info
      do j=1,nev
         if(flg_newweight .and. .not. flg_rwl_add) then
c            call pwhgnewweight(iunin,iunout)
            write(*,*) ' dead feature! exiting ...'
            call exit(-1)
         elseif(flg_rwl_add) then
c     read num_stored_evt from input .lhe file
            call lhefreadev(iunin)
            count_stored_evt = count_stored_evt + 1
            if(num_old_weights > 0) then
               weights(1:num_old_weights,count_stored_evt) =
     1              rwl_weights(1:num_old_weights)
            endif
            call rwl_handle_lhe('put',num_stored_evt,count_stored_evt)
         else
            call pwhgevent
            if(nup.eq.0) then
               write(*,*) ' nup = 0 skipping event'
               cycle
            endif
c     store current event and increase stored count
            count_stored_evt = count_stored_evt + 1
            if(num_stored_evt == 0) then
               call lhefwritev(iunout)
               call dotestplots
               count_stored_evt = 1
            else
               call rwl_handle_lhe('put',
     1              num_stored_evt,count_stored_evt)
            endif
         endif
         if(num_stored_evt /= 0 .and.
     1        (num_stored_evt == count_stored_evt .or. j==nev)) then
c     save current parameters
            call rwl_setup_params_weights(0)
c     save random status, not to alter the sequence of generated events
c     due to the addition of weights
            call randomsave
c     retrieve events and add weights
            do count_weights = num_old_weights+1, num_weights
c     setup parameters for current weight
               call rwl_setup_params_weights(count_weights)
               do k_stored_evt=1,count_stored_evt
c     we could have used 'get';
c     with the 'getweight' option only the information neede for computing a new
c     weight is retrieved from the event, saving time ...
                  call rwl_handle_lhe
     1                 ('get',count_stored_evt,k_stored_evt)
                  call rwl_compute_new_weight
     1                 (weights(count_weights,k_stored_evt))
               enddo
c     restore current parameters
               call rwl_setup_params_weights(-1)
            enddo
            call randomrestore
c     cache full random status before call to next event. This call does not
c     alter the sequence of random number generator, but it provide a starting
c     point to reach faster subsequent random number.
            call cachefullrandomstatus
            do k_stored_evt=1,count_stored_evt
c copy stored event to LH common block
               call rwl_handle_lhe('get',count_stored_evt,k_stored_evt)
c     Write event and weights
               rwl_weights(:) = weights(:,k_stored_evt)
               call lhefwritev(iunout)
c               write(iunout,*) '# new weight ',weights(:,k_stored_evt)
               call dotestplots
            enddo
            count_stored_evt = 0
         endif
      enddo
      if (flg_nlotest) then
         if(rnd_cwhichseed.eq.'none') then
            filename=pwgprefix(1:lprefix)//
     1           'alone-output'
         else
            filename=pwgprefix(1:lprefix)//
     1           'alone-output'//trim(rnd_cwhichseed)
         endif
         call pwhgsetout
         call pwhgtopout(filename)
      endif
      if(flg_newweight) then
            write(*,*) ' dead feature!'
            write(*,*) ' exiting ...'
            call exit(-1)
         call pwhg_io_close(iunin)
         call pwhg_io_close(iunout)
      else
         call lhefwritetrailer(iunout)
         call pwhg_io_close(iunout)
      endif
 999  continue
      call write_counters
c this causes powheginput to print all unused keywords
c in the powheg.input file; useful to catch mispelled keywords
      tmp=powheginput('print unused tokens')
      contains
      subroutine dotestplots
      if(testplots) then
         call lhtohep
         call analysis(xwgtup)
         call pwhgaccumup
         if (mod(j,5000).eq.0) then
            if(rnd_cwhichseed.eq.'none') then
               filename=pwgprefix(1:lprefix)//
     1              'pwhgalone-output'
            else
               filename=pwgprefix(1:lprefix)//
     1              'pwhgalone-output'//trim(rnd_cwhichseed)
            endif
            call pwhgsetout
            call pwhgtopout(filename)
         endif
      endif
      end subroutine
      end

      subroutine rwl_compute_new_weight(weight)
      implicit none
      real * 8 weight
      include 'nlegborn.h'
      include 'pwhg_flst.h'
      include 'pwhg_rad.h'
      include 'pwhg_flg.h'
      include 'pwhg_rwl.h'
      include 'LesHouches.h'
      real * 8 newweight
      logical pwhg_isfinite
      integer iii,mcalls,ncalls
      real * 8 xxx(ndiminteg)
      call setrandom(rwl_seed,rwl_n1,rwl_n2)
      mcalls=0
      ncalls=0
      if(rwl_type.eq.1) then
c     generate an event with a single try (no hit and miss)
         call genwrapper('btilde',2,mcalls,ncalls,xxx,iii)
         call mintwrapper_result('btilde',rwl_index,newweight)
         if(flg_fullrwgt) then
             rad_ubornsubp = rwl_index
             call fullrwgt(newweight)
         endif
      elseif(rwl_type.eq.2) then
         call genwrapper('remn',2,mcalls,ncalls,xxx,iii)
         call mintwrapper_result('remn',rwl_index,newweight)
      elseif(rwl_type.eq.3) then
         call genwrapper('reg',2,mcalls,ncalls,xxx,iii)
         call mintwrapper_result('reg',rwl_index,newweight)
      else
         write(*,*) 'Error in pwhgnewweight, invalid rad_type: ',
     $        rwl_type
         call exit(-1)
      endif
      if(.not.pwhg_isfinite(newweight)) newweight=0d0
      weight = xwgtup * newweight/rwl_weight
      end

      subroutine openoutputrw(iunrwgt)
      implicit none
      include 'pwhg_flg.h'
      include 'pwhg_rnd.h'
      integer iunrwgt
      character * 20 pwgprefix
      character * 200 file
      integer lprefix,iret
      common/cpwgprefix/pwgprefix,lprefix
      real * 8 powheginput
      logical exist
      if(rnd_cwhichseed.ne.'none') then
         file=pwgprefix(1:lprefix)//'events-rwgt-'
     1        //trim(rnd_cwhichseed)//'.lhe'
      else
         file=pwgprefix(1:lprefix)//'events-rwgt.lhe'
      endif
      if(powheginput("#clobberlhe").ne.1) then
         inquire(file=file,exist=exist)
         if(exist) then
            write(*,*) 'pwhg_main: error, file ',trim(file),
     1           ' exists! will not overwrite, exiting ...'
            call exit(-1)
         endif
      endif
      call pwhg_io_open_write(trim(file),iunrwgt,
     1     flg_compress_lhe,iret)
      if(iret /= 0) then
         write(*,*) ' could not open ',trim(file),' for writing'
         write(*,*) ' exiting ...'
         call exit(-1)
      endif
      end
      

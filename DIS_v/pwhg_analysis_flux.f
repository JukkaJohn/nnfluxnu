!  The next subroutines, open some histograms and prepare them 
!      to receive data 
!  You can substitute these  with your favourite ones
!  init   :  opens the histograms
!  topout :  closes them
!  pwhgfill  :  fills the histograms with data

      subroutine init_hist
         implicit none
         include  'LesHouches.h'
         include 'pwhg_math.h'
         include 'nlegborn.h'
         include 'pwhg_kn.h'
         call inihists
         
         call bookupeqbins('sigtot',1d0,0d0,1d0)

         call bookupeqbins('sigfaserv',1d0,0d0,1d0)
         call bookupeqbins('sigfaserv2',1d0,0d0,1d0)
         call bookupeqbins('sigsnde',1d0,0d0,1d0)
         call bookupeqbins('sigsndmu',1d0,0d0,1d0)
         call bookupeqbins('sigflaree',1d0,0d0,1d0)
         call bookupeqbins('sigflaremu',1d0,0d0,1d0)

         call bookupeqbins('Q',  0.5d0, 2d0, 20d0)

         call bookupeqbins('Q2', (2000d0-25d0)/40d0, 25d0, 2000d0)
         call bookupeqbins('yadQ2', (310d0-10d0)/30d0, 10d0, 310d0)
         call bookupeqbins('El', (2000d0-25d0)/40d0, 25d0, 2000d0)
         call bookupeqbins('Eh', (2000d0-25d0)/40d0, 25d0, 2000d0)
         call bookupeqbins('Enu', (6000d0-25d0)/120d0, 25d0, 6000d0)
         call bookupeqbins('tantheta', 0.001d0, 0d0, 0.05d0)
C         call bookupeqbins('theta', 0.001d0, 0d0, 0.5d0)
         call bookupeqbins('Epion', (2000d0-25d0)/40d0, 25d0, 2000d0)
         call bookupeqbins('ptpion', 0.5d0, 0d0, 15d0)
         call bookupeqbins('hitrack', 5d0, 0d0, 300d0)
         call bookupeqbins('Esumpion', (2000d0-25d0)/40d0, 25d0, 2000d0)
         call bookupeqbins('ptsumpion', 0.5d0, 0d0,15d0)
         call bookupeqbins('neutronE', (4000d0-25d0)/40d0, 25d0, 4000d0)
         call bookupeqbins('kaonE', (4000d0-25d0)/40d0, 25d0, 4000d0)
         call bookupeqbins('kaonEL', (4000d0-25d0)/40d0, 25d0, 4000d0)
C        add this plot between lepton and any hadron, pion stable.   
         call bookupeqbins('minaziangle', 0.05d0, 0d0, 3.20d0)
         call bookupeqbins('etapion', 0.001d0, 0d0, 0.030d0)
         call bookupeqbins('etapionsum', 0.001d0, 0d0, 0.030d0)
         call bookupeqbins('y',  (0.95d0-0.04d0)/20d0 , 0.04d0, 0.95d0)
         call bookupeqbins('x',  0.05d0 , 0d0, 1d0)
         call bookupeqbins('xl',  0.05d0 , 0d0, 1d0)   
         call bookupeqbins('ptl', 2d0, 0d0, 1d0)
         call bookupeqbins('etal',0.1d0,0d0,5d0)
         call bookupeqbins('ptj_overQ_breit', 0.01d0, 0d0, 0.5d0)
C        add distributions for the comparison to faser data from 2403.12520
         call bookupeqbins('multiplicity', 1d0, 0d0,100d0)
         call bookupeqbins('dphi',10d0,0d0,180d0) 
         call bookupeqbins('ptlepton',100d0,0d0,2000d0)
         call bookupeqbins('theta', 0.005d0, 0d0, 0.1d0)

      end
       
      subroutine analysis(dsig0)
         implicit none
         real * 8 dsig0
         include 'nlegborn.h'
         include 'hepevt.h'
         include 'pwhg_math.h' 
         include 'LesHouches.h'
         include 'pwhg_kn.h'
         include 'pwhg_weights.h'
         ! include 'pwhg_rad.h'
         ! include 'pwhg_rwl.h'
         logical ini
         data ini/.true./
         save ini
         real * 8 dsig(weights_max)
   
         integer   maxtrack,maxjet
         integer i,j,njets
         parameter (maxtrack=2048,maxjet=128)
         integer ipion(maxtrack),ilepton(maxtrack),iphoton(maxtrack)
         integer icharged(maxtrack)
         integer nlepton,nphoton,ncp
         ! we need to tell to this analysis file which program is running it
         character * 6 WHCPRG
         common/cWHCPRG/WHCPRG
         ! data WHCPRG/'NLO   '/
         integer nlep, npartons,nunident                         !lepton, quark, lepton initialstate, parton initialstate 
         real * 8 plep(4,maxtrack), plephard(4),pquark(4,maxtrack),plis(4),ppis(4),q(4)              !momenta of lepton, quark, lepton initialstate, parton is, momentum transfere
         real * 8 El_hardest,Eh,theta,theta2
         real * 8 plab(0:3,maxtrack), pjets(0:3,maxjet)
         real * 8 y,x,Q2,xl               !xs. inv mass of incoming parton-electron, momentum transfer variable Q2=-q^2
         real * 8 ptj(maxjet),yj(maxjet),phij(maxjet), eta1, ptjetmin, absEtaMax
         real * 8 ptl, etal
         real * 8, external :: phepdot, eta, kt2
         real * 8 refin(1:4), refout(1:4)
         real * 8 alpha, beta, ptbreit
         real * 8, save :: Eproton, Elepton, sbeams
         logical, save :: fixed_target
         logical,save :: recombination
         real *8, external :: powheginput
         integer npion
         real * 8 Epion_hardest,thetapion,ppion_hardest(4),ptpion,ptmax
         real * 8 ppion(4,maxtrack),ptpionsum,Epionsum,ppionsum(4)
         real * 8 ypion,etapion,etapionsum,masspion
         real * 8 minaziangle,dphi
         real * 8 nchargedparticles
         integer nkaon,nneutron,nkaonL
         real * 8 pkaon(4,maxtrack),pneutron(4,maxtrack),
     1            pkaonL(4,maxtrack)
         real * 8 Ekaon,EkaonL,Eneutron
         real * 8 dylph,detalph,dphilph,drlph
         real * 8 factor,MW,GF
         real * 8 psum(4),ylep,etalep,ptlep,masslep
         real * 8 vec(3)
         integer * 4 ntracks05,ntracks01
         
         interface 
            subroutine getdphi(p1,p2,dphi,phi1,phi2)
            real(kind=8),intent(in)   :: p1(4),p2(4)
            real(kind=8),intent(out)  :: dphi
            real(kind=8),intent(out),optional :: phi1,phi2
            end subroutine

            logical function charged(pid) 
               integer :: pid
            end function charged
         end interface
       
        
         GF=1.16637D-5
         MW=80.385d0

         dsig=0d0
         call multi_plot_setup(dsig0,dsig, weights_max)
   
         if(ini) then         
            if(whcprg.eq.'NLO') then
               write(*,*) '       NLO analysis'
            elseif(WHCPRG.eq.'LHE   ') then
               write(*,*) '       LHE analysis'
            elseif(WHCPRG.eq.'HERWIG') then
               write (*,*) '           HERWIG ANALYSIS            '
            elseif(WHCPRG.eq.'PYTHIA') then
               write (*,*) '           PYTHIA ANALYSIS            '
            endif
            ini=.false.

            Elepton = powheginput("ebeam1")
            Eproton = powheginput("ebeam2")
            fixed_target = (powheginput("#fixed_target") .eq. 1d0)
            recombination = (powheginput("#recombination") .eq. 1d0)


            if(fixed_target) then
               sbeams = 2*Elepton*Eproton + Eproton**2
            else
               sbeams = 4 * Elepton * Eproton
            endif


            if(whcprg.eq.'NLO') then
               if( abs( 4d0 * ebmup(1) * ebmup(2) - sbeams) .gt. 1d-4)
     $              then
                  write(*,*) "inconsistency in the calculation of ",
     $                 "the partonic center-of-mass energy in the ",
     $                 "analysis, aborting ..."
                  stop
               endif
            endif
            
         endif

         nlep = 0
         npartons = 0
         plep = 0d0
         ppis = 0d0
         plis = 0d0
         pquark = 0d0
         nunident = 0
         npion=0
         ppion=0d0
         ppionsum=0d0
         ppion_hardest=0d0
         Epion_hardest=0d0
         Epionsum=0d0
         ptpion=0d0
         ptpionsum=0d0
         nchargedparticles=0d0
         ncp=0
         nkaon=0
         nkaonL=0
         nneutron=0
         Ekaon=0d0
         EkaonL=0d0
         Eneutron=0d0
         nlepton=0
         nphoton=0

         if(whcprg.eq.'NLO   '.or.whcprg.eq.'LHE   '
     $        .or.whcprg.eq.'PYTHIA') then
            do i=1,nhep
               !if(idhep(i).eq.11) print*, phep(1:4,i), eta(phep(:,i)), sqrt(kt2(phep(:,i)))
   !     Initial states
               if(isthep(i).eq.-1.or.isthep(i).eq.21) then
                  if(abs(idhep(i)).le.16 .and.abs(idhep(i)).ge.11 ) then
                     plis(1:4) = phep(1:4,i)
                  else if(abs(idhep(i)).le.5.or.idhep(i).eq.21) then
                     ppis(1:4) = phep(1:4,i)
                  else
                     stop 'Wrong initial state'
                  endif
   !     Final states
               else if(isthep(i).eq.1) then
                  if(charged(idhep(i)))then
                     nchargedparticles=nchargedparticles+1d0
                     ncp=ncp+1
                     icharged(ncp)=i
                     psum=psum+phep(1:4,i)
                  end if
                  if(idhep(i).eq.22)then
                     nphoton=nphoton+1
                     iphoton(nphoton)=i
                  end if
                  if(abs(idhep(i)).eq.11.or.abs(idhep(i)).eq.13.or.
     1               abs(idhep(i)).eq.15)then
                     nlepton=nlepton+1
                     ilepton(nlepton)=i
                  end if
                  if(abs(idhep(i)).eq.311)then ! Kaon
                     nkaon=nkaon+1
                     pkaon(:,nkaon)=phep(1:4,i)
                  end if 
                  if(abs(idhep(i)).eq.130)then ! K_L
                     nkaonL=nkaonL+1
                     pkaonL(:,nkaonL)=phep(1:4,i)
                  end if
                  if(abs(idhep(i)).eq.2112)then
                     nneutron=nneutron+1
                     pneutron(:,nneutron)=phep(1:4,i)
                  end if
                  if(abs(idhep(i)).ge.11 .and. abs(idhep(i)).le.16) then
                     nlep = nlep + 1
                     plep(1:4,nlep) = phep(1:4,i)
                  elseif (abs(idhep(i)) <= 9 .or. abs(idhep(i)) == 21 .or. abs(idhep(i)) > 100 ) then 
                     npartons = npartons + 1
                     pquark(1:4,npartons) = phep(1:4,i)
                     if(abs(idhep(i)).eq.111.or.abs(idhep(i)).eq.211)then
                        npion=npion+1
                        ppion(1:4,npion)=phep(1:4,i)
                        ipion(npion)=i
                     end if
                  else
                     nunident = nunident + 1
                     ! print*, 'idhep(i)', idhep(i)
                  endif
               endif
            enddo
         else
            print*, 'Wrong program', whcprg
            stop
         endif

         if (nunident>0) print*, 'nunident', nunident
   
         if(nlep<1) stop 'Wrong number of leptons'
   
         El_hardest = 0d0
         do i = 1, nlep
            if (plep(4,i) > El_hardest ) then
               El_hardest = plep(4,i)
               plephard(1:4) = plep(1:4,i) 
            endif
         end do
         
! Momentum of the highest momentum track.         
         if(ncp.gt.1)then
            call sortbypt(ncp,icharged(ncp))
         endif
         ptmax=DSQRT(phep(1,icharged(1))**2+phep(2,icharged(1))**2)

C cuts for vertex selection from 2403.12520
!         ntracks05=0
!         ntracks01=0
!         do i=1,ncp
!            theta=datan(dsqrt(phep(1,icharged(i))**2+phep(2,icharged(i))
!     1                        **2)/phep(3,icharged(i)))
!            if(dtan(theta).le.0.5d0)then
!               ntracks05=ntracks05+1
!            end if
!            if(dtan(theta).le.0.1d0)then
!               ntracks01=ntracks01+1
!            end if
!         end do
!C -------------------------------------------------

         if(nlepton.gt.1) call sortbypt(nlepton,ilepton(1:nlepton))
! Dressed lepton.
         if (recombination) then
            do i=1,nlepton
               do j=1,nphoton
                  if (abs(phep(4,iphoton(j))).gt.0) then 
                     call getdydetadphidr(phep(1:4,ilepton(i)),
     1                 phep(1:4,iphoton(j)),dylph,detalph,dphilph,drlph)
                     if (drlph.lt.0.2d0) then
                        phep(1:4,ilepton(i)) = phep(1:4,ilepton(i))
     1                       + phep(1:4,iphoton(j))
                        phep(1:4,iphoton(j)) = 0d0
                     endif
                  endif
               enddo            
            enddo         
            do i=2,nlepton
               if (abs(phep(4,ilepton(i))).gt.0) then
                  call getdydetadphidr(phep(1:4,ilepton(1)),
     1                 phep(1:4,ilepton(i)),dylph,detalph,dphilph,drlph)
                  if (drlph.lt.0.2d0) then
                     phep(1:4,ilepton(1)) = phep(1:4,ilepton(1))
     1                + phep(1:4,ilepton(i))
                     phep(1:4,ilepton(i))=0d0
                  endif 
               endif
            enddo
         endif
         El_hardest=phep(4,ilepton(1))
         plephard(1:4)=phep(1:4,ilepton(1))

         Eh=0d0
         minaziangle=HUGE(1d0)
         do i=1,npartons
            Eh=Eh+pquark(4,i)
            CALL getdphi(plephard,pquark(1:4,i),dphi)
            IF(dphi.LT.minaziangle) minaziangle=dphi
         end do
! Kaon,neutron
         do i=1,nkaon
            if(pkaon(4,i).gt.Ekaon) Ekaon=pkaon(4,i)
         end do
         do i=1,nkaonL
            if(pkaonL(4,i).gt.EkaonL) EkaonL=pkaonL(4,i)
         end do
         do i=1,nneutron
            if(pneutron(4,i).gt.Eneutron) Eneutron=pneutron(4,i)
         end do
! PION STUFF

         if(npion.gt.1) call sortbypt(npion,ipion(1:npion))
         if(npion.ge.1)then
            Epion_hardest=phep(4,ipion(1))
            ppion_hardest=phep(1:4,ipion(1))
            do i=1,npion
               ppionsum(1:4)=ppionsum(1:4)+ppion(1:4,i)
               Epionsum=Epionsum+ppion(4,i)
            end do
            call getyetaptmass(ppion_hardest,ypion,etapion,ptpion,
     1                         masspion)
            call getyetaptmass(ppionsum,ypion,etapionsum,ptpionsum,
     1                         masspion)
         end if
   
         q(:) = plis(:) - plephard(:)

         xl = plis(4)/ebmup(1)
         
         Q2 = phepdot(q,q)
         Q2 = -Q2


!         theta=2d0*DASIN(DSQRT(Q2/4d0/(Eh+El_hardest)/El_hardest))
         theta=DATAN(DSQRT(plephard(1)**2+plephard(2)**2)/plephard(3))
         IF(theta.GT.pi/2d0) theta=pi-theta
!     Q^2 = xl xdis y S
!     x = Q^2/(S*y*xdis)
!     y = (q.P)/(l/in.P)
         
         y = phepdot(ppis,q) / phepdot(ppis,plis)      
         x = Q2 / (sbeams * y * xl)
    
         
         call filld('sigtot',0.5d0,dsig)   

         ptl  = sqrt(kt2(plephard(:)))
         call getrapidity(plephard(:),etal)
         call filld('ptl', ptl, dsig)
         call filld('etal', etal, dsig)
C     Comparison to real FASERv data
C     compute the angle between the final state lepton and the sum of
C     all other particles
         call getdphi(plephard,psum,dphi)
         call getyetaptmass(plephard,ylep,etalep,ptlep,masslep)


!         if(El_hardest.ge.100d0)then
!         end if
!         if(Eh.ge.100d0)then
!         end if
!         if(DTAN(theta).le.0.025d0)then
!            call filld("sigsndmu",0.5d0,dsig)
!         end if
         if(El_hardest.ge.100d0.and.Eh.ge.100d0.and.DTAN(theta).le.0.025)then
            call filld("sigfaserv",0.5d0,dsig)                         
            call filld('Q', sqrt(Q2), dsig)
            call filld('Q2', Q2, dsig)
            call filld('x', x, dsig)
            call filld('y', y, dsig)
            call filld('xl', xl, dsig)
            call filld('El', El_hardest, dsig)
            call filld('Eh', Eh, dsig)
            call filld('Enu',plis(4), dsig)
            call filld('tantheta', DTAN(theta), dsig)
            call filld('theta', theta, dsig)
            call filld('minaziangle',minaziangle,dsig)
            if(npion.ge.1)then
               call filld('Epion',Epion_hardest,dsig)
               call filld('ptpion',ptpion,dsig)
               call filld('Esumpion',Epionsum,dsig)
               call filld('ptsumpion',ptpionsum,dsig)
               call filld('etapion',etapion,dsig)
               call filld('etapionsum',etapionsum,dsig)
            end if
            call filld('multiplicity',nchargedparticles,dsig)
            call filld('neutronE',Eneutron,dsig)
            call filld('kaonE',Ekaon,dsig)
            call filld('kaonEL',EkaonL,dsig)
            call filld('hitrack',ptmax,dsig)
            dphi=dphi/pi*180d0
            call filld('dphi',dphi,dsig)
            call filld('ptlepton',ptlep,dsig)
         end if
         if(El_hardest.ge.100d0.and.Eh.ge.100d0.and.DTAN(theta).le.0.5)then
            call filld("sigfaserv2",0.5d0,dsig)
         end if
         ! SND e
         if(El_hardest.ge.20d0.and.Eh.ge.20d0.and.theta.le.0.5)then
            call filld("sigsnde",0.5d0,dsig)
         end if
         ! SND mu
         if(El_hardest.ge.20d0.and.Eh.ge.20d0.and.theta.le.0.15)then
         end if
         if(El_hardest.ge.2d0.and.Eh.ge.2d0.and.theta.le.0.5.and.Eh.le.2000d0)then
            call filld("sigflaree",0.5d0,dsig)
         end if
         if(El_hardest.ge.2d0.and.Eh.ge.2d0.and.theta.le.0.025)then
            call filld("sigflaremu",0.5d0,dsig)
         end if


         if(npartons == 2) then
            refin  = x *ebmup(2) * (/ 0d0, 0d0, -1d0, 1d0/)
            if(fixed_target) refin = refin/2d0 ! The fake massless beam has E = mp/2d0, hence the factor 2
            if(plis(3) < 0) refin(3)=-refin(3) ! Flip the sign of z if necessary
            refout = q(:) + refin
            
            alpha = 2d0 * phepdot(pquark(:, 1), refout)/Q2
            beta  = 2d0 * phepdot(pquark(:, 1), refin)/Q2
            
            ptbreit = sqrt(alpha*beta)
         else
            ptbreit = 0d0
         endif
         call filld('ptj_overQ_breit', ptbreit, dsig)

         
         

         
c$$$         print *, "Check event!"
c$$$         print *, "npartons = ", npartons
c$$$         print *, "Q2 = ", Q2, 2* phepdot(refin, refout)
c$$$         print *, "refout.m2 = ", phepdot(refout,refout) 
  
      end
  
      subroutine getdydetadphidr(p1,p2,dy,deta,dphi,dr)
      implicit none
      include 'pwhg_math.h' 
      real * 8 p1(*),p2(*),dy,deta,dphi,dr
      real * 8 y1,eta1,pt1,mass1,phi1
      real * 8 y2,eta2,pt2,mass2,phi2
      call getyetaptmass(p1,y1,eta1,pt1,mass1)
      call getyetaptmass(p2,y2,eta2,pt2,mass2)
      dy=y1-y2
      deta=eta1-eta2
      phi1=atan2(p1(1),p1(2))
      phi2=atan2(p2(1),p2(2))
      dphi=abs(phi1-phi2)
      dphi=min(dphi,2d0*pi-dphi)
      dr=sqrt(deta**2+dphi**2)
      end
  
  
      function phepDot(p_A,p_B)
         implicit none
         real * 8  phepDot
         real * 8  p_A(4),p_B(4)
         phepDot=p_A(4)*p_B(4)-p_A(1)*p_B(1)-p_A(2)*p_B(2)-p_A(3)*p_B(3)
      end
  
      function kt2(p)
         implicit none
         real * 8 kt2, p(1:4)
   
         kt2 = p(1)**2 + p(2)**2
      end
  
      function eta(p)
         implicit none
         real * 8 eta, p(0:3), normp, norm
   
         normp = norm(p)
         if(normp.gt.p(3)) then
            eta = 0.5d0 * log((normp + p(3)) / (normp - p(3)))
         else
            eta = sign(1d100,p(3)) 
         endif
      end
  
      function norm(p)
         implicit none
         real * 8 norm, p(0:3)
   
         norm = p(1)**2 + p(2)**2 + p(3)**2
         norm = sqrt(max(0d0,norm))
         end
   
         function getrapidity0(p)
         implicit none
         real * 8 p(0:3),getrapidity0
         getrapidity0=0.5d0*log((p(0)+p(3))/(p(0)-p(3)))
      end
  
      subroutine getrapidity(p,y)
         implicit none
         real * 8 p(4),y
         y=0.5d0*log((p(4)+p(3))/(p(4)-p(3)))
      end
  
      function azi(p)
         implicit none
         real * 8 pi
         parameter(pi = 3.141592653589793D0)
         real * 8 azi,p(0:3)
         azi = atan(p(2)/p(1))
         if (p(1).lt.0d0) then
            if (azi.gt.0d0) then               
               azi = azi - pi
            else
               azi = azi + pi
            endif
         endif    
      end
        
      ! Builds pp like jets. Should take as argument all final state
      ! partons/hadrons. Returns the jets pt ordered. 
      subroutine buildjets(n,pin,pj,njets,ptj,yj,phij, ptjetmin, absEtaMax)
         implicit none
         integer n
         double precision pin(0:3,n) 
         integer maxtrack,maxjet
         parameter (maxtrack=2048,maxjet=128)
   
   !     Output
         double precision pj(0:3,maxjet)
   !     Internal
         integer mu, njets, njets_all, ntracks, ijet, j
         integer jetvec(maxtrack)
         double precision pjet(4,maxjet), pjet_all(4,maxjet)
         double precision ptrack(4,maxtrack)
         double precision ptj(maxjet),yj(maxjet),phij(maxjet), ptjetmin, absEtaMax
         double precision ptj_all(maxjet),yj_all(maxjet),phi_all(maxjet)
         double precision R, ptmin_fastkt, palg
         double precision, external :: azi, eta
   
   
         ptrack = 0d0
         jetvec = 0
         pjet = 0d0
         pjet_all = 0d0
         njets=0
         njets_all = 0
         ntracks = n
         
         ptrack(4,1:n)=pin(0,1:n)
         ptrack(1:3,1:n)=pin(1:3,1:n)
         
         R = 0.8d0
         ptmin_fastkt = 0d0
         palg = -1d0
   !      palg = 0d0
   !      palg = -1d0
c     -1 is anti_kt, +1 is kt
  
         call fastjetppgenkt(ptrack,ntracks,r,palg,ptmin_fastkt,pjet,njets,jetvec)

         if(njets.gt.maxjet) then
            print*, 'njets out of bounds!!', njets, maxjet
            stop
         endif
         
         j=0
         pjet_all=pjet
         njets_all=njets
         pjet=0d0
         njets=0d0
         ptj = 0d0
         yj = 0d0
         
         do ijet=1,njets_all
            ptj_all(ijet) = sqrt(pjet_all(1,ijet)**2 + pjet_all(2,ijet)**2)
            call getrapidity(pjet_all(:,ijet),yj_all(ijet))
            phi_all(ijet) = azi(pjet_all(:,ijet))
  
          if(ptj_all(ijet).gt.ptjetmin.and.
     $        abs(eta(cshift(pjet_all(:,ijet),-1))).lt.absEtaMax) then
      !     if(ptj_all(ijet).gt.ptalljetmin.and.
      ! $        abs(yj_all(ijet)).lt.yjetmax) then
               j=j+1
               pjet(:,j)=pjet_all(:,ijet)
               ptj(j) = ptj_all(ijet)
               yj(j) = yj_all(ijet)
               phij(j) = phi_all(ijet)
          endif
         enddo
         njets=j
   
         pj = 0d0
   
         do ijet=1,njets
            do mu=1,3
               pj(mu,ijet)=pjet(mu,ijet)
            enddo
            pj(0,ijet)=pjet(4,ijet)
         enddo
              
      end
  
      subroutine getyetaptmass(p,y,eta,pt,mass)
      implicit none
      real * 8 p(*),y,eta,pt,mass,pv
      y=0.5d0*log((p(4)+p(3))/(p(4)-p(3)))
      pt=sqrt(p(1)**2+p(2)**2)
      pv=sqrt(pt**2+p(3)**2)
      eta=0.5d0*log((pv+p(3))/(pv-p(3)))
      mass=sqrt(abs(p(4)**2-pv**2))
      end

      subroutine sortbypt(n,iarr)
      implicit none
      integer n,iarr(n)
      include 'hepevt.h'
      integer j,k
      real * 8 tmp,pt(nmxhep)
      logical touched
      do j=1,n
         pt(j)=sqrt(phep(1,iarr(j))**2+phep(2,iarr(j))**2)
      enddo
c bubble sort
      touched=.true.
      do while(touched)
         touched=.false.
         do j=1,n-1
            if(pt(j).lt.pt(j+1)) then
               k=iarr(j)
               iarr(j)=iarr(j+1)
               iarr(j+1)=k
               tmp=pt(j)
               pt(j)=pt(j+1)
               pt(j+1)=tmp
               touched=.true.
            endif
         enddo
      enddo
      end
  
      subroutine getdphi(p1,p2,dphi,phi1,phi2)
      implicit none
      include 'pwhg_math.h' 
      real(kind=8),intent(in)   :: p1(4),p2(4)
      real(kind=8),intent(out)  :: dphi
      real(kind=8),intent(out),optional :: phi1,phi2
      real(kind=8) :: phiA, phiB,pt1,pt2

      pt1=sqrt(p1(1)**2+p1(2)**2)
      pt2=sqrt(p2(1)**2+p2(2)**2)
      
      if(p1(2).ge.0)then
         phiA = dacos(p1(1)/pt1)
      else
         phiA=2*pi-dacos(p1(1)/pt1)
      end if 
      if(p2(2).ge.0) then
         phiB = dacos(p2(1)/pt2)
      else
         phiB = 2*pi - dacos(p2(1)/pt2)
      end if 
      dphi=abs(phiA - phiB)
      if(dphi.gt.pi) dphi=2*pi-dphi
      if(present(phi1).and.present(phi2))then
         phi1=phiA 
         phi2=phiB
      end if 
      end subroutine getdphi

      logical function charged(pid) 
         implicit none
         integer :: pid
         integer,dimension(121),parameter :: cpid=(/1,2,3,4,5,6,11,13,15,
     1 24,
     1    211,9000211,100211,10211,9010211,213,10213,20213,9000213,100,213,
     1    9010213,9040213,215,10215,9000215,9010215,217,9000217,9010217,219,
     1    321,9000321,10321,100321,9010321,9020321,323,10323,20323,100323,
     1    9000323,30323,325,9000325,10325,20325,9010325,9020325,327,9010327,
     1    329,9000329,
     1    411,10411,413,10413,20413,415,431,10431,433,10433,10433,20433,435,
     1    521,10521,523,10523,20523,525,541,543,10541,10543,20543,545
     1    2212,2224,2214,1114,3222,3112,3224,3114,3312,3314,3334,4122,4222,4212,4224,
     1    4214,4232,4322,4324,4412,4422,4414,4424,4432,4434,4444,5112,5222,5114,5224,
     1    5132,5312,5314,5332,5334,5242,5422,5424,5444,5512,5514,5532,5534,
     1    5554/)

         charged=.FALSE.
         if(any(cpid.eq.abs(pid)))charged=.TRUE.
      end function charged

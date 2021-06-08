     

 

          PROGRAM MAIN
!          use procs127, only : TEST_R_I, PYTHON_SET_UP_RECBIS 
          IMPLICIT NONE 
!         integer, parameter :: nx_dg=16, ny_dg=16
!         integer, parameter :: nx=10, ny=10, nz=1
!         integer, parameter :: nx_dg=2, ny_dg=2
         integer, parameter :: nx_dg=7, ny_dg=7
!         integer, parameter :: nx=3, ny=3, nz=1
         integer, parameter :: nx=5, ny=5, nz=1
         integer, parameter :: nscalar=1, ndim=2, ntime=1

         real dx_dg,dy_dg, dx,dy, x_wide_dg,y_wide_dg, x_wide,y_wide, xloc(3),yloc(3), ddx(2),block_x_start(2)
         INTEGER nonods_dg, nonods, nloc, ele,k, i,j,ii, kloc, totele, node
         integer ireturn_zeros_outside_grid
         real, allocatable :: x_all(:,:), value_grid(:,:,:,:,:), value_mesh(:,:,:) 
         integer, allocatable :: x_ndgln(:)


! ordering of the triangle elements (nx=2,ny=2): 
! 3  2     6   9  8      12    + 12 to each
! 1     4  5   7     10  11    + 12 to each
! 3  2     6   9  8      12
! 1     4  5   7     10  11
           print *,'here1'
           nloc=3
           totele=nx_dg*ny_dg*2
           nonods_dg=totele*nloc
!           nonods2=(nx+1)*(ny+1)
           allocate(x_all(2,nonods_dg) )
           allocate(x_ndgln(nonods_dg) ) 
           x_wide_dg=1.0
           y_wide_dg=1.0
           dx_dg=x_wide_dg/nx_dg
           dy_dg=y_wide_dg/ny_dg
           do j=1,ny_dg
              do i=1,nx_dg
                 do ii=1,2
                    ele=2*((j-1)*nx_dg + i -1) +ii
                    if(ii==1) then
                       xloc(1) = dx_dg*(i-1) 
                       xloc(2) = dx_dg*(i-1) + dx_dg
                       xloc(3) = dx_dg*(i-1) 
                       yloc(1) = dy_dg*(j-1) 
                       yloc(2) = dy_dg*(j-1) + dy_dg
                       yloc(3) = dy_dg*(j-1) + dy_dg
                    else
                       xloc(1) = dx_dg*(i-1) 
                       xloc(2) = dx_dg*(i-1) + dx_dg
                       xloc(3) = dx_dg*(i-1) + dx_dg
                       yloc(1) = dy_dg*(j-1) 
                       yloc(2) = dy_dg*(j-1) 
                       yloc(3) = dy_dg*(j-1) + dy_dg
                    endif
                    do kloc=1,nloc
                       k = (ele-1)*nloc + kloc
                       x_ndgln(k)=k
                       x_all(1,k)=xloc(kloc)
                       x_all(2,k)=yloc(kloc)
!                       coordinates(k,3)=0.0
!                       print *,'ele,kloc,k,coords:',ele,kloc,k,coordinates(k,1),coordinates(k,2)
                    end do
                 end do
              end do
           end do

           print *,'x_all(1,:):',x_all(1,:)
           print *,'x_all(2,:):',x_all(2,:)
           print *,'x_ndgln:',x_ndgln

!          print *,'inside'
!           x_wide=1.0
!           y_wide=1.0
           x_wide=1.5
           y_wide=1.5
           dx=x_wide/(nx-1)
           dy=y_wide/(ny-1)
           nonods=nonods_dg
           ddx(1)=dx
           ddx(2)=dy
!           block_x_start=0.0
           block_x_start=0.25
!           block_x_start=1.e-5
           
           allocate(value_grid(nscalar,nx,ny,nz,ntime), value_mesh(nscalar,nonods,ntime) )
           value_mesh=1.0
           value_mesh(1,:,1)=x_all(1,:) 
!           value_mesh(1,:,1)=3.14
!           do j=1,ny
!              do i=1,nx
!                 nod=(j-1)*nx_dg + i 
!              end do
!           end do
          ireturn_zeros_outside_grid=0

!          call simple_interpolate_from_mesh_to_grid_one(value_grid, value_mesh, x_all,x_ndgln,ddx,block_x_start, &
          call simple_interpolate_from_mesh_to_grid(value_grid, value_mesh, x_all,x_ndgln,ddx,block_x_start, &
                                 nx,ny,nz, ireturn_zeros_outside_grid, totele,nloc,nonods,nscalar,ndim,ntime) 

          print *,'value_grid:',value_grid

          call interpolate_from_grid_to_mesh(value_mesh, value_grid, block_x_start, ddx, x_all, &
                                      ireturn_zeros_outside_grid,nscalar, nx,ny,nz, nonods,ndim,ntime) 

          do node=1,nonods
             print *,node,x_all(1,node),value_mesh(1,node,1)
          end do

          STOP
          END PROGRAM MAIN




      subroutine simple_interpolate_from_mesh_to_grid_old(vg,vm,x,xgn,ddx,start,nx,ny,nz, &
                         ireturn_zeros_outside_grid,nEl,nloc,nNodes, nscalar,ndim,ntime) 
! *******************************************************************************************************
! This sub interpolates value_grid from value_mesh.
! It is a simplified interface to the subroutine interpolate_from_mesh_to_grid. 
! value_grid is on the structured mesh used for subdomain rom.
! value_mesh is on the unstructured mesh. 
! x_all(idim,node) = coordinates of node node and this has dimensions ndim do 1st index is the dimension number. 
! x_ndgln((ele-1)*nloc+iloc) = global node number for the coordinates for element ele and local node number (for an element ele) iloc
! ddx(1:ndim) = the width of the cells in the x,y,z-directions. 
! block_x_start(1:ndim)= start coordinates of the structured block.
! nx,ny,nz dimensions of the structured block system; nz has to be at least equal to 1 eg for 2d problems. 
! if(ireturn_zeros_outside_grid.ne.0) ! if the point is outside the mesh then return a zero.
! nonods=number of nodes.
! totele=number of elements.
! nloc=number of local nodes in each element.
! nscalar=number of scalar values we are interpolating.
! ndim=number of dimensiuons.
! ntime=number of time steps. 
! 
! the phthon interface looks like: 
! value_grid = simple_interpolate_from_mesh_to_grid(value_mesh, 
!                                         x_all,x_ndgln,ddx,block_x_start,nx,ny,nz, 
!                                         nonods,totele,nloc,nscalar,ndim,ntime) 
! *******************************************************************************************************
      implicit none
      integer, intent(in) :: nx,ny,nz,nNodes,ireturn_zeros_outside_grid,nEl,nloc,nscalar,ndim,ntime
      real, intent(out) :: vg(nscalar,nx,ny,nz,ntime)
!      real, intent(out) :: l1234(nloc,nx,ny,nz) 
!      integer, intent(out) :: elewic(nx,ny,nz)
      real, intent(in) :: vm(nscalar,nNodes,ntime)
!      integer, intent(in) :: igot_ele_store
!      real, intent(in) :: l1234_keep(nloc,nx,ny,nz) 
!      integer, intent(in) :: elewic_keep(nx,ny,nz)
      integer, intent(in) :: xgn(nloc*nEl) 
      real, intent(in) :: x(ndim,nNodes) 
      real, intent(in) :: ddx(ndim), start(ndim) 
! local variables...
      integer igot_ele_store
      real, allocatable :: l1234(:,:,:,:),  l1234_keep(:,:,:,:)
      integer, allocatable :: elewic(:,:,:),  elewic_keep(:,:,:)

! more local variables
      integer nonods, totele
      real, allocatable :: value_grid(:,:,:,:,:)
      real, allocatable :: value_mesh(:,:,:)
      real, allocatable :: x_all(:,:) 
      integer, allocatable :: x_ndgln(:)
      real, allocatable :: block_x_start(:)

      allocate(l1234(nloc,nx,ny,nz), l1234_keep(nloc,nx,ny,nz)) 
      allocate(elewic(nx,ny,nz), elewic_keep(nx,ny,nz)) 

      allocate(value_grid(nscalar,nx,ny,nz,ntime) )
      allocate(value_mesh(nscalar,nNodes,ntime) )
      allocate(x_all(ndim,nNodes) )
      allocate(x_ndgln(nloc*nEl) )
      allocate(block_x_start(ndim) )

      nonods = nNodes  
      totele = nEl
      value_mesh = vm
      x_all = x
      x_ndgln = xgn
      block_x_start = start

!      print *, nonods, totele

! elewic(i,j,k)= element number that the cell i,j,k is in
! l1234(iloc,i,j,k) = local coordinates for local node number iloc and cell i,j,k 
! igot_ele_store = says we have stored values in elewic and l1234 
! elewic_keep,l1234_keep are the same as  elewic,l1234 but with stored values of these.
      igot_ele_store=0
      call interpolate_from_mesh_to_grid(value_grid, elewic,l1234,value_mesh, &
              ireturn_zeros_outside_grid,igot_ele_store,elewic_keep,l1234_keep, &
                                         x_all,x_ndgln,ddx,block_x_start,nx,ny,nz, &
                                         totele,nloc,nonods,nscalar,ndim,ntime) 

      vg(:,:,:,:,:) = value_grid(:,:,:,:,:)
      return 
      end subroutine simple_interpolate_from_mesh_to_grid_old




! in python: value_grid = u2r.simple_interpolate_from_mesh_to_grid(value_mesh,x_all,x_ndgln,ddx,block_x_start,nx,ny,nz,nEl,nloc,nNodes,nscalar,ndim,nTime) 
      subroutine simple_interpolate_from_mesh_to_grid(value_grid, value_mesh, x_all,x_ndgln,ddx,block_x_start, &
                              nx,ny,nz, ireturn_zeros_outside_grid, totele,nloc,nonods,nscalar,ndim,ntime) 
! *******************************************************************************************************
! This sub interpolates value_grid from value_mesh.
! It is a simplified interface to the subroutine interpolate_from_mesh_to_grid. 
! value_grid is on the structured mesh used for subdomain rom.
! value_mesh is on the unstructured mesh. 
! x_all(idim,node) = coordinates of node node and this has dimensions ndim do 1st index is the dimension number. 
! x_ndgln((ele-1)*nloc+iloc) = global node number for the coordinates for element ele and local node number (for an element ele) iloc
! ddx(1:ndim) = the width of the cells in the x,y,z-directions. 
! block_x_start(1:ndim)= start coordinates of the structured block.
! nx,ny,nz dimensions of the structured block system; nz has to be at least equal to 1 eg for 2d problems. 
! if(ireturn_zeros_outside_grid.ne.0) ! if the point is outside the mesh then return a zero.
! nonods=number of nodes.
! totele=number of elements.
! nloc=number of local nodes in each element.
! nscalar=number of scalar values we are interpolating.
! ndim=number of dimensiuons.
! ntime=number of time steps. 
! 
! the phthon interface looks like: 
! value_grid = simple_interpolate_from_mesh_to_grid(value_mesh, 
!                                         x_all,x_ndgln,ddx,block_x_start,nx,ny,nz, 
!                                         nonods,totele,nloc,nscalar,ndim,ntime) 
! *******************************************************************************************************
      implicit none
      integer, intent(in) :: nx,ny,nz,nonods,totele,nloc,nscalar,ndim,ntime, ireturn_zeros_outside_grid
      real, intent(out) :: value_grid(nscalar,nx,ny,nz,ntime)
!      real, intent(out) :: l1234(nloc,nx,ny,nz) 
!      integer, intent(out) :: elewic(nx,ny,nz)
      real, intent(in) :: value_mesh(nscalar,nonods,ntime)
!      integer, intent(in) :: igot_ele_store
!      real, intent(in) :: l1234_keep(nloc,nx,ny,nz) 
!      integer, intent(in) :: elewic_keep(nx,ny,nz)
      integer, intent(in) :: x_ndgln(nloc*totele) 
      real, intent(in) :: x_all(ndim,nonods) 
      real, intent(in) :: ddx(ndim), block_x_start(ndim) 
! local variables...
      integer igot_ele_store
      real, allocatable :: l1234(:,:,:,:),  l1234_keep(:,:,:,:)
      integer, allocatable :: elewic(:,:,:),  elewic_keep(:,:,:)
      allocate(l1234(nloc,nx,ny,nz), l1234_keep(nloc,nx,ny,nz)) 
      allocate(elewic(nx,ny,nz), elewic_keep(nx,ny,nz)) 

! elewic(i,j,k)= element number that the cell i,j,k is in
! l1234(iloc,i,j,k) = local coordinates for local node number iloc and cell i,j,k 
! igot_ele_store = says we have stored values in elewic and l1234 
! elewic_keep,l1234_keep are the same as  elewic,l1234 but with stored values of these.
      igot_ele_store=0
      call interpolate_from_mesh_to_grid(value_grid, elewic,l1234,value_mesh, &
              ireturn_zeros_outside_grid,igot_ele_store,elewic_keep,l1234_keep, &
                                         x_all,x_ndgln,ddx,block_x_start,nx,ny,nz, &
                                         totele,nloc,nonods,nscalar,ndim,ntime) 
      return 
      end subroutine simple_interpolate_from_mesh_to_grid




      subroutine interpolate_from_mesh_to_grid(value_grid, elewic,l1234,value_mesh, &
                  ireturn_zeros_outside_grid,igot_ele_store,elewic_keep,l1234_keep, &
                                         x_all,x_ndgln,ddx,block_x_start,nx,ny,nz, &
                                         totele,nloc,nonods,nscalar,ndim,ntime) 
! *******************************************************************************************************
! This sub interpolates value_grid from value_mesh.
! value_grid is on the structured mesh used for subdomain rom.
! value_mesh is on the unstructured mesh. 
! if(ireturn_zeros_outside_grid.ne.0) ! if the point is outside the mesh then return a zero.
! elewic(i,j,k)= element number that the cell i,j,k is in
! l1234(iloc,i,j,k) = local coordinates for local node number iloc and cell i,j,k 
! igot_ele_store = says we have stored values in elewic and l1234 
! elewic_keep,l1234_keep are the same as  elewic,l1234 but with stored values of these.
! x_all(idim,node) = coordinates of node node and this has dimensions ndim do 1st index is the dimension number. 
! x_ndgln((ele-1)*nloc+iloc) = global node number for the coordinates for element ele and local node number (for an element ele) iloc
! ddx(1:ndim) = the width of the cells in the x,y,z-directions. 
! block_x_start(1:ndim)= start coordinates of the structured block.
! nx,ny,nz dimensions of the structured block system; nz has to be at least equal to 1 eg for 2d problems. 
! nonods=number of nodes.
! totele=number of elements.
! nloc=number of local nodes in each element.
! nscalar=number of scalar values we are interpolating.
! ndim=number of dimensiuons.
! ntime=number of time steps. 
! 
! the phthon interface looks like: 
! value_grid, elewic, l1234 = interpolate_from_mesh_to_grid(value_mesh, 
!                                         ireturn_zeros_outside_grid,igot_ele_store,elewic_keep,l1234_keep, 
!                                         x_all,x_ndgln,ddx,block_x_start,nx,ny,nz, 
!                                         nonods,totele,nloc,nscalar,ndim,ntime) 
! *******************************************************************************************************
      implicit none
      integer, intent(in) :: nx,ny,nz,nonods,totele,nloc,nscalar,ndim,ntime
      integer, intent(in) :: ireturn_zeros_outside_grid
      real, intent(out) :: value_grid(nscalar,nx,ny,nz,ntime)
      real, intent(out) :: l1234(nloc,nx,ny,nz) 
      integer, intent(out) :: elewic(nx,ny,nz)
      real, intent(in) :: value_mesh(nscalar,nonods,ntime)
      integer, intent(in) :: igot_ele_store
      real, intent(in) :: l1234_keep(nloc,nx,ny,nz) 
      integer, intent(in) :: elewic_keep(nx,ny,nz)
      integer, intent(in) :: x_ndgln(nloc*totele) 
      real, intent(in) :: x_all(ndim,nonods) 
      real, intent(in) :: ddx(ndim), block_x_start(ndim) 
! local variables...
      real toler,infiny
      logical test ! if test then perform a test of the element oriantation. 
      parameter(toler=1.0e-10,infiny=1.e+10, test=.true.) 
      real, allocatable :: mincork(:,:,:), dist2(:,:,:)
      real dx,dy,dz, xpt(ndim), loccords(nloc), loccords2(nloc), rsum
      real block_start_x, block_start_y, block_start_z, dist2ele
!      real min_x(ndim), max_x(ndim) 
      real ele_min_x_all(ndim), ele_max_x_all(ndim), loc_x_all(ndim,nloc), pos(ndim) 
!      real rnnx(ndim) 
      integer nnx(ndim), istart(3),ifinish(3), locnods(nloc)
      integer ele,idim,i,j,k, iloc
!      integer jump, idist2_keep, kk3(3),jj3(3),ii3(3), kk,jj,ii, k2,j2,i2, idist2, iii,jjj,kkk
      integer jump, idist2_keep, kk3(3),jj3(3),ii3(3), k2,j2,i2, idist2, iii,jjj,kkk
      integer icount,count,  iiicount, jjjcount
      real mincor
!      real TRI_tet_LOCCORDS ! real function
      logical got_ele_store, return_zeros_outside_grid
      
      got_ele_store=.not.(igot_ele_store==0) 
      return_zeros_outside_grid = .not.(ireturn_zeros_outside_grid==0)

      dx=ddx(1); dy=ddx(2); if(ndim==3) dz=ddx(3)
      nnx(1)=nx; nnx(2)=ny; if(ndim==3) nnx(3)=nz
!         rnnx(:)=real(nnx(:))
      block_start_x=block_x_start(1); block_start_y=block_x_start(2); if(ndim==3) block_start_z=block_x_start(3)
!
!      xc_all(:) = x_all
      if(test) then ! teset the oriantation of the nodes in an element - the 1st element. 
         do ele=1,totele ! check all elements have the correct oriantation. 
            locnods(1:nloc) = x_ndgln((ELE-1)*NLOC+1:(ELE-1)*NLOC+NLOC)
            loc_x_all(:,:) = x_all(:,LOCNODS(:))
            do idim=1,ndim
               xpt(idim) = sum(loc_x_all(idim,:))/real(nloc) 
            end do
            CALL TRI_tet_LOCCORDS(xpt, loccords,loc_x_all,NDIM,NLOC) 
            if(minval(loccords(:))<0.0) stop 8211 ! the oriantation of the element is wrong. 
         end do
      endif

      if(.not.got_ele_store) then 
         allocate(mincork(nx,ny,nz), dist2(nx,ny,nz) )
         mincork(:,:,:)=-infiny
         dist2(:,:,:) =infiny
         elewic=0

!         min_x(:)=block_x_start(:)
!         max_x(:)=block_x_start(:) + ddx(:) * real( nnx(:) - 1 )
         istart(3)=1 ! in case its 2d.
         ifinish(3)=1 ! in case its 2d.

         do ele=1,totele
            locnods(1:nloc)=x_ndgln((ELE-1)*NLOC+1:(ELE-1)*NLOC+NLOC)
            loc_x_all(:,:) = x_all(:,LOCNODS(:))
! Remember xpt(1) = block_x_start(1) + ddx(1)*real(i-1) so re-arrange to get the below
            do idim=1,ndim
               ele_max_x_all(idim)=maxval(x_all(idim,LOCNODS(:)))
               ele_min_x_all(idim)=minval(x_all(idim,LOCNODS(:)))
!               istart(idim) =int( 0.00001+ real(nnx(idim)+1)*(ele_min_x_all(idim)-min_x(idim))/ (max_x(idim) - min_x(idim)))  - 1
!               ifinish(idim)=int( 0.00001+ real(nnx(idim)+1)*(ele_max_x_all(idim)-min_x(idim))/ (max_x(idim) - min_x(idim)))  + 1
               istart(idim) =int( (ele_min_x_all(idim)-block_x_start(idim))/ ddx(idim))  + 1        -1! final no is adjustment for safety
               ifinish(idim)=int( (ele_max_x_all(idim)-block_x_start(idim))/ ddx(idim))  + 1        +1! final no is adjustment for safety
               istart(idim)= max(1, min( istart(idim),  nnx(idim)-1 ) ) 
               ifinish(idim)=max(2, min( ifinish(idim), nnx(idim)   ) )
            end do
!            print *,'ele:',ele
!            print *,'istart(2),ifinish(2):',istart(2),ifinish(2)
!            print *,'istart(1),ifinish(1):',istart(1),ifinish(1)
! 
            do k=istart(3),ifinish(3)
            do j=istart(2),ifinish(2)
            do i=istart(1),ifinish(1)
               xpt(1) = block_x_start(1) + ddx(1)*real(i-1)
               xpt(2) = block_x_start(2) + ddx(2)*real(j-1)
   if(ndim==3) xpt(3) = block_x_start(3) + ddx(3)*real(k-1) 
               CALL TRI_tet_LOCCORDS(xpt, loccords,loc_x_all,NDIM,NLOC) 

               mincor = minval( loccords(:) )

               if((mincor<0.0).and.(mincork(i,j,k)<0.0)) then 
! store distance...
                  loccords2(:)=max(0.0, loccords(:) ) 
                  rsum = sum( loccords2(:) )
                  loccords2(:)=loccords2(:) / max(toler, rsum)  ! has sum to 1 property. 

                  pos(:)=0.0
                  do iloc=1,nloc
                     pos(:)= pos(:) + loccords2(iloc) * loc_x_all(:,iloc) 
                  end do
                  dist2ele =sum( (pos(:)-xpt(:))**2 )
                  if(dist2ele < dist2(i,j,k)) then
                     dist2(i,j,k)=dist2ele
                     mincork(i,j,k)=mincor
                     elewic(i,j,k)=ele
                     l1234(:,i,j,k) = loccords(:)
                  endif
               else if(mincor > mincork(i,j,k)) then ! find the element the pt is in or nearest element
                  mincork(i,j,k)=mincor
                  elewic(i,j,k)=ele
                  l1234(:,i,j,k) = loccords(:)
               endif


!               if((i==3).and.(j==2)) then
!               if((ele==3).or.(ele==8)) then
!               if(ele==3) then
!                  print *,'ele=',ele
!                  print *,'mincor,i,j,xpt(1),xpt(2):',mincor,i,j,xpt(1),xpt(2)
!                  print *,'loccords:',loccords
!                  print *,'loc_x_all(1,:):',loc_x_all(1,:)
!                  print *,'loc_x_all(2,:):',loc_x_all(2,:)
!               endif
!               endif

            end do
            end do
            end do
         end do ! do ele=1,totele
! 
!       print *,'totele:',totele
         iiicount=0
         jjjcount=0
         do k=1,nz
         do j=1,ny
         do i=1,nx
            iiicount=iiicount+1
            if(elewic(i,j,k)==0) then ! adopt nearest value
               jjjcount=jjjcount+1
! find nearest i2,j2,k2 to i,j,k that has a none zero elewic(i2,j2,k2) 
               idist2_keep = (nx+1)**2 + (ny+1)**2 + (nz+1)**2
               icount=0
               count=0
               do jump=1,max(nx,ny,nz) 

                  kk3(1)=max(k-jump,1); kk3(2)=k; kk3(3)=min(k+jump,nz)
                  jj3(1)=max(j-jump,1); jj3(2)=j; jj3(3)=min(j+jump,ny)
                  ii3(1)=max(i-jump,1); ii3(2)=i; ii3(3)=min(i+jump,nx)
! k: 
               if(ndim==3) then
                  do kkk=1,3 
                  k2=kk3(kkk)  
                  do j2=jj3(1),jj3(3)  
                  do i2=ii3(1),ii3(3)

                     ele = elewic(i2,j2,k2)
                     if(ele.ne.0) then
                        icount=1
                        idist2 = (i-i2)**2 + (j-j2)**2 + (k-k2)**2
                        if(idist2 < idist2_keep) then
                           elewic(i,j,k)=ele
                           idist2_keep=idist2
                        endif
                     endif

                  end do
                  end do
                  end do
               endif
! j:
                  do k2=kk3(1),kk3(3)
                  do jjj=1,3
                  j2=jj3(jjj)  
                  do i2=ii3(1),ii3(3)

                     ele = elewic(i2,j2,k2)
                     if(ele.ne.0) then
                        icount=1
                        idist2 = (i-i2)**2 + (j-j2)**2 + (k-k2)**2
                        if(idist2 < idist2_keep) then
                           elewic(i,j,k)=ele
                           idist2_keep=idist2
                        endif
                     endif

                  end do
                  end do
                  end do
! i: 
                  do k2=kk3(1),kk3(3)
                  do j2=jj3(1),jj3(3)  
                  do iii=1,3
                  i2=ii3(iii) 

                     ele = elewic(i2,j2,k2)
                     if(ele.ne.0) then
                        icount=1
                        idist2 = (i-i2)**2 + (j-j2)**2 + (k-k2)**2
                        if(idist2 < idist2_keep) then
                           elewic(i,j,k)=ele
                           idist2_keep=idist2
                        endif
                     endif

                  end do
                  end do
                  end do

                  count=count+icount

!         if((i==5).and.(j==1)) then
!                  print *,'jump,ele,elewic(i,j,k):',jump,ele,elewic(i,j,k)
!         endif

                  if((count.ge.4).and.(elewic(i,j,k).ne.0)) exit
               end do

               ele = elewic(i,j,k)
!               print *,'ele,totele,i,j,k:',ele,totele,i,j,k
!               print *,'x_ndgln((ELE-1)*NLOC+1:(ELE-1)*NLOC+NLOC):',x_ndgln((ELE-1)*NLOC+1:(ELE-1)*NLOC+NLOC)
               locnods(1:nloc) = x_ndgln((ELE-1)*NLOC+1:(ELE-1)*NLOC+NLOC)
               loc_x_all(:,:) = x_all(:,LOCNODS(:))
               xpt(1) = block_x_start(1) + ddx(1)*real(i-1)
               xpt(2) = block_x_start(2) + ddx(2)*real(j-1)
   if(ndim==3) xpt(3) = block_x_start(3) + ddx(3)*real(k-1)
               CALL TRI_tet_LOCCORDS(xpt, loccords,loc_x_all,NDIM,NLOC) 
! store distance...
               l1234(:,i,j,k) = loccords(:)

            endif ! if(elewic(i,j,k)==0) then 
         end do
         end do
         end do
         
!       print *,'elewic:',elewic
!       print *,'elewic(2,1,1), elewic(3,1,1):',elewic(2,1,1), elewic(3,1,1)
         if(minval(elewic)==0) stop 5663 ! there is a problem as not all the grid cells are assigned elements
!         stop 7
      else ! f(.NOT.GOT_ELE_NO) then 
         elewic=elewic_keep
         l1234 =l1234_keep
      endif ! if(.NOT.GOT_ELE_NO) then else

!      print *,'iiicount, jjjcount:',iiicount, jjjcount
                
! iterpolate from mesh to structured grid or block. 
      do k=1,nz
         do j=1,ny
            do i=1,nx
!               xpt(1) = block_x_start(1) + ddx(1)*real(i-1)
!               xpt(2) = block_x_start(2) + ddx(2)*real(j-1)
!   if(ndim==3) xpt(3) = block_x_start(3) + ddx(3)*real(k-1)
                   
               ele=elewic(i,j,k) 
               loccords(:)=l1234(:,i,j,k) 
               if(return_zeros_outside_grid) then ! if the point is outside the mesh then return a zero.
                  if(minval(loccords(:)).lt.toler) loccords(:)=0.0 
               endif
! make an adjustment in case the point is outside of an element...
               loccords(:)=max(0.0, loccords(:) ) 
               rsum = sum( loccords(:) )
               loccords(:)=loccords(:) / max(toler, rsum)  ! has sum to 1 property.

               locnods(1:nloc)=X_NDGLN((ELE-1)*NLOC+1:(ELE-1)*NLOC+NLOC)

               value_grid(:,i,j,k,:) =0.0
               do iloc=1,nloc
                  value_grid(:,i,j,k,:) = value_grid(:,i,j,k,:) + loccords(iloc) * value_mesh(:,locnods(iloc),:) 
               end do
!               if(value_grid(1,i,j,k,1)>0.0001) then
!                  print *,'loccords:',loccords
!                  print *,'value_mesh(:,locnods(:),:):',value_mesh(:,locnods(:),:)
!                  print *,'value_grid(:,i,j,k,:):',value_grid(:,i,j,k,:)
!                  stop 2821
!               endif

            end do
         end do
      end do

      end subroutine interpolate_from_mesh_to_grid
! 
! 
! 
! in python:
! value_remesh = u2r.interpolate_from_grid_to_mesh(value_grid, block_x_start, ddx, x_all, ireturn_zeros_outside_mesh,nscalar,nx,ny,nz,nNodes,ndim,nTime)
      subroutine interpolate_from_grid_to_mesh(value_mesh, value_grid, block_x_start, ddx, x_all, &
                                   ireturn_zeros_outside_mesh, nscalar, nx,ny,nz, nonods,ndim,ntime) 
! *******************************************************************************************************
! This sub interpolates value_mesh from value_grid.
! value_grid is on the structured mesh used for subdomain rom.
! value_mesh is on the unstructured mesh. 
! elewic(i,j,k)= element number that the cell i,j,k is in
! l1234(iloc,i,j,k) = local coordinates for local node number iloc and cell i,j,k 
! igot_ele_store = says we have stored values in elewic and l1234 
! elewic_keep,l1234_keep are the same as  elewic,l1234 but with stored values of these.
! x_all(idim,node) = coordinates of node node and this has dimensions ndim do 1st index is the dimension number. 
! if(ireturn_zeros_outside_grid.ne.0) ! if the point is outside the mesh then return a zero.
! ddx(1:ndim) = the width of the cells in the x,y,z-directions. 
! block_x_start(1:ndim)= start coordinates of the structured block.
! if(ireturn_zeros_outside_mesh.ne.0) then return a zero for mesh nodes outside of the grid interpolation. 
! nx,ny,nz dimensions of the structured block system.
! nonods=number of nodes.
! nscalar=number of scalar values we are interpolating.
! ndim=number of dimensiuons.
! ntime=number of time steps. 
! 
! the phthon interface looks like: 
! value_mesh = interpolate_from_grid_to_mesh(value_grid, block_x_start, ddx, x_all, nscalar,nx,ny,nz,nonods,ndim)
! *******************************************************************************************************
      implicit none 
      integer, intent(in) :: nx,ny,nz,nonods,ndim,nscalar,ntime
      integer, intent(in) :: ireturn_zeros_outside_mesh
      real, intent(out) :: value_mesh(nscalar,nonods,ntime)
      real, intent(in) :: value_grid(nscalar,nx,ny,nz,ntime)
      real, intent(in) :: block_x_start(ndim), ddx(ndim), x_all(ndim,nonods)
! local variables...
      real toler,bigger_toler
      parameter(toler=1.0e-10,bigger_toler=1.0e-7) 
      integer node, i,j,k, ii,jj,kk, nnx(ndim), idim
      real wi(2,ndim), w_loc(2,2,ndim-1), xpt(ndim) 
      logical return_zeros_outside_mesh, pt_outside_grid

      return_zeros_outside_mesh = .not.(ireturn_zeros_outside_mesh==0)
!      print *, 'return_zeros_outside_mesh', return_zeros_outside_mesh

      nnx(1)=nx; nnx(2)=ny; if(ndim==3) nnx(3)=nz

!      print *,'block_x_start:',block_x_start
!      print *,'ddx:',ddx
!      print *,'nnx:',nnx
      w_loc = 0.0 ! may need to do for 2D. 

      do node=1,nonods
         xpt(1:ndim)=x_all(1:ndim,node) 
         if(return_zeros_outside_mesh) then
            pt_outside_grid = .false.
            do idim=1,ndim
               if(      (xpt(idim) < block_x_start(idim)-bigger_toler) &
                  .or. (xpt(idim) > block_x_start(idim)+nnx(idim)*ddx(idim) + bigger_toler) ) &
                     pt_outside_grid = .true.
            end do
            
            if( pt_outside_grid ) then
               !print *, pt_outside_grid, xpt
               value_mesh(:, node, :) =0.0
               cycle
            endif
         endif
!         print *,'node,xpt:',node,xpt

! Remember  xpt(1) = block_x_start(1) + ddx(1)*real(i-1) so re-arrange to get the below
         i = int( (xpt(1)-block_x_start(1))/ddx(1) ) +1 
         i= max(1, min( i,  nnx(1)-1 ) ) 
         j = int( (xpt(2)-block_x_start(2))/ddx(2) ) +1 
         j= max(1, min( j,  nnx(2)-1 ) ) 
      if(ndim==3) then
         k = int( (xpt(3)-block_x_start(3))/ddx(3) ) +1 
         k= max(1, min( k,  nnx(3)-1 ) ) 
      else
         k= 1
      endif

         wi(1,1) = (xpt(1)- (block_x_start(1) + ddx(1)*(i-1) ) )/ ddx(1)
         wi(1,2) = (xpt(2)- (block_x_start(2) + ddx(2)*(j-1) ) )/ ddx(2)
      if(ndim==3) then
         wi(1,3) = (xpt(3)- (block_x_start(3) + ddx(3)*(k-1) ) )/ ddx(3)
      endif
         wi(1,:) = max(0.0,   min( 1.0, wi(1,:) )   )
         wi(1,:)=1.0 - wi(1,:) 

         wi(2,:)=1.0 - wi(1,:) 
         do kk=1,ndim-1
         do jj=1,2
         do ii=1,2
      if(ndim==3) then
            w_loc(ii,jj,kk)=wi(ii,1)*wi(jj,2)*wi(kk,3) 
      else
            w_loc(ii,jj,kk)=wi(ii,1)*wi(jj,2)
      endif
         end do
         end do
         end do
         w_loc=w_loc/max(sum(w_loc), toler) 
!         print *,'i,j:',i,j
!         print *,'w_loc(1,:,1):', w_loc(1,:,1)
!         print *,'w_loc(2,:,1):', w_loc(2,:,1)

         value_mesh(:, node, :) =0.0
         do kk=1,ndim-1
         do jj=1,2
         do ii=1,2
!            print *,'i+ii-1, j+jj-1, k+kk-1:',i+ii-1, j+jj-1, k+kk-1
!            print *,'(xpt(1)-block_x_start(1))/ddx(1):',(xpt(1)-block_x_start(1))/ddx(1)
            value_mesh(:, node,:) = value_mesh(:, node,:) + w_loc(ii,jj,kk)*value_grid(:, i+ii-1, j+jj-1, k+kk-1, :)  
         end do
         end do
         end do
      end do
      end subroutine interpolate_from_grid_to_mesh
! 
! 

! 
        SUBROUTINE TRI_tet_LOCCORDS(Xpt, LOCCORDS, X_CORNERS_ALL, NDIM,CV_NLOC)
! obtain the local coordinates LOCCORDS from a pt in or outside the tet/triangle Xpt
! with corner nodes X_CORNERS_ALL
            IMPLICIT NONE
            INTEGER, intent(in) :: NDIM,CV_NLOC
            REAL, dimension(NDIM), intent(in) :: Xpt
            REAL, dimension(NDIM+1), intent(inout) :: LOCCORDS
            REAL, dimension(NDIM,CV_NLOC), intent(in) :: X_CORNERS_ALL

            IF (NDIM==3) THEN

                CALL TRILOCCORDS(Xpt(1),Xpt(2),Xpt(3), &
                    LOCCORDS(1),LOCCORDS(2),LOCCORDS(3),LOCCORDS(4),&
! The 4 corners of the tet...
                    X_CORNERS_ALL(1,1),X_CORNERS_ALL(2,1),X_CORNERS_ALL(3,1),&
                    X_CORNERS_ALL(1,2),X_CORNERS_ALL(2,2),X_CORNERS_ALL(3,2),&
                    X_CORNERS_ALL(1,3),X_CORNERS_ALL(2,3),X_CORNERS_ALL(3,3),&
                    X_CORNERS_ALL(1,4),X_CORNERS_ALL(2,4),X_CORNERS_ALL(3,4) )
            ELSE
                CALL TRILOCCORDS2D(Xpt(1),Xpt(2), &
                    LOCCORDS(1),LOCCORDS(2),LOCCORDS(3),&
!     The 3 corners of the tri...
                    X_CORNERS_ALL(1,1),X_CORNERS_ALL(2,1),&
                    X_CORNERS_ALL(1,2),X_CORNERS_ALL(2,2),&
                    X_CORNERS_ALL(1,3),X_CORNERS_ALL(2,3) )
            END IF
            ! From  the local coordinates find the shape function value...
            RETURN
        END SUBROUTINE TRI_tet_LOCCORDS
!    
!    
!    
!     	
    !sprint_to_do!think about this
    Subroutine TRILOCCORDS(Xp,Yp,Zp,N1, N2, N3, N4, X1,Y1,Z1, X2,Y2,Z2, X3,Y3,Z3, X4,Y4,Z4  )

        IMPLICIT NONE
        Real Xp, Yp, Zp

        Real N1, N2, N3, N4

        Real X1,Y1,Z1
        Real X2,Y2,Z2
        Real X3,Y3,Z3
        Real X4,Y4,Z4

        Real Volume
        Real TetVolume

        !     calculate element volume...

        Volume = TetVolume(X1, Y1, Z1, &
            X2, Y2, Z2, &
            X3, Y3, Z3, &
            X4, Y4, Z4)

        Volume = Volume /6.0


        !     vol coords...

        N1 = TetVolume(Xp, Yp, Zp, &
            X2, Y2, Z2, &
            X3, Y3, Z3, &
            X4, Y4, Z4)

        N1 = N1/(6.0*Volume)



        N2 = TetVolume(X1, Y1, Z1, &
            Xp, Yp, Zp, &
            X3, Y3, Z3, &
            X4, Y4, Z4)

        N2 = N2/(6.0*Volume)



        N3 = TetVolume(X1, Y1, Z1, &
            X2, Y2, Z2, &
            Xp, Yp, Zp, &
            X4, Y4, Z4)

        N3 = N3/(6.0*Volume)


        N4 = TetVolume(X1, Y1, Z1, &
            X2, Y2, Z2, &
            X3, Y3, Z3, &
            Xp, Yp, Zp)

        N4 = N4/(6.0*Volume)


        Return

    end subroutine triloccords
! 
    !

   function tetvolume(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3)
      IMPLICIT NONE

     real, intent(in) :: x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3

     real :: tetvolume

     ! tetvolume = 1.0 / 6.0 * det |three tet edge vectors|
     ! Chris' tets have a clockwise base, hence the sign change in the det
     tetvolume = &
       (  &
         & - (x1 - x0) * ((y2 - y0) * (z3 - z0) - (y3 - y0) * (z2 - z0)) &
         & + (y1 - y0) * ((x2 - x0) * (z3 - z0) - (x3 - x0) * (z2 - z0)) &
         & - (z1 - z0) * ((x2 - x0) * (y3 - y0) - (x3 - x0) * (y2 - y0)) &
       & ) / 6.0

   end function tetvolume
! 
    !
    !
    Subroutine TRILOCCORDS2D(Xp,Yp, N1, N2, N3, X1,Y1, X2,Y2, X3,Y3 )

        IMPLICIT NONE
        Real Xp,Yp, &
            N1, N2, N3,  &
            X1,Y1, &
            X2,Y2, &
            X3,Y3

        Real AREA
        Real TRIAREAF_SIGN

        AREA = TRIAREAF_SIGN( X1, Y1, X2, Y2, X3, Y3)
        !     area coords...

        N1 = TRIAREAF_SIGN(Xp, Yp,  &
            &     X2, Y2,  &
            &     X3, Y3 )

        N1 = N1/AREA



        N2 = TRIAREAF_SIGN(X1, Y1, &
            &     Xp, Yp,  &
            &     X3, Y3 )

        N2 = N2/AREA



        N3 = TRIAREAF_SIGN(X1, Y1,  &
            &     X2, Y2,  &
            &     Xp, Yp )

        N3 = N3/AREA


        Return

    end subroutine triloccords2d
! 
! 

  real function triareaf_SIGN( x1, y1, x2, y2, x3, y3 )
    implicit none
    real :: x1, y1, x2, y2, x3, y3

    triareaf_SIGN = 0.5 * ( ( x2 * y3 - y2 * x3 ) - x1 * ( y3 - y2 ) + y1 * ( x3 - x2 ) )

    return
  end function triareaf_SIGN



  !subroutine CrossProduct( cp, a, b )
  !  implicit none
  !  real, dimension( : ), intent( inout ) :: cp
  !  real, dimension( : ), intent( in ) :: a, b!

  !  cp( 1 ) = a( 2 ) * b( 3 ) - a( 3 ) * b( 2 )
  !  cp( 2 ) = a( 3 ) * b( 1 ) - a( 1 ) * b( 3 )
  !  cp( 3 ) = a( 1 ) * b( 2 ) - a( 2 ) * b( 1 )

  !  return
  !end subroutine CrossProduct


 ! subroutine CrossProduct(cp, a, b, n, m, l)
 !   implicit none
 !   integer, intent(in) :: n, m, l
 !   real , intent(out) :: cp(l)
 !   real, intent(in) :: a(n)
 !   real, intent(in) :: b(m)
 !   cp( 1 ) = a( 2 ) * b( 3 ) - a( 3 ) * b( 2 )
 !   cp( 2 ) = a( 3 ) * b( 1 ) - a( 1 ) * b( 3 )
 !   cp( 3 ) = a( 1 ) * b( 2 ) - a( 2 ) * b( 1 )
 !   !return
 ! end subroutine CrossProduct


!    
!    

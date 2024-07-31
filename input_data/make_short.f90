program make_short_all

  implicit none
  CHARACTER*150 :: filename
  INTEGER       :: file_counter
  DOUBLE PRECISION :: lamb_min, lamb_max, mol_mass
  LOGICAL       :: write_wlen
  double precision, parameter :: amu = 1.66053892d-24

  ! Initial setup
  write_wlen = .TRUE.
  file_counter = 1

  ! Read in wlen boundaries
  open(unit=40, file='short_stream_lambs_mass.dat')
  read(40,*)
  read(40,*) lamb_min
  read(40,*)
  read(40,*) lamb_max
  read(40,*)
  read(40,*) mol_mass
  close(40)

  mol_mass = mol_mass*amu

  ! Loop through files to shorten them
  open(unit=40, file='sigma_list.ls')

  do while (1 > 0)

     read(40,*,end=20) filename
     write(*,*) trim(adjustl(filename))

     if (file_counter > 1) then
        write_wlen = .FALSE.
     end if

     call make_short(filename, lamb_min, &
          lamb_max, write_wlen, mol_mass)

     file_counter = file_counter + 1

  end do
20 close(40)  

end program make_short_all

subroutine make_short(filename, lamb_min, &
     lamb_max, write_wlen, mol_mass)

  implicit none

  integer :: lamb_len
  double precision :: lamb_min, lamb_max, mol_mass
  double precision, allocatable :: lamb(:), opa(:)
  character*150 :: filename
  logical       :: write_wlen

  !write(*,*)
  !write(*,*) 'Getting opa dim...'
  call get_opa_dim(lamb_len, filename)
  !write(*,*) '    Done.'

  allocate(lamb(lamb_len))
  allocate(opa(lamb_len))
  
  !write(*,*)
  !write(*,*) 'Reading opacity...'
  call read_wlen_opa(lamb, opa, lamb_len, &
       filename)
  !write(*,*) '    Done.'

  opa = opa / mol_mass
  
  !write(*,*)
  !write(*,*) 'Writing opacity...'
  call write_opa(lamb, opa, lamb_min, lamb_max, &
       lamb_len, 'short_stream/'//filename, write_wlen)
  !write(*,*) '    Done.'
  
  deallocate(lamb)
  deallocate(opa)

end subroutine make_short

subroutine write_opa(lamb, opacity, lamb_min, lamb_max, &
     lamb_len, filename, write_wlen)

  implicit none

  CHARACTER*150                 :: filename
  INTEGER                       :: lamb_len, i_lamb
  DOUBLE PRECISION              :: lamb_min, lamb_max
  DOUBLE PRECISION              :: lamb(lamb_len), opacity(lamb_len)
  LOGICAL                       :: write_wlen

  ! To overwrite file if it exsists
  open(unit=10,file=trim(adjustl(filename)))
  write(10,*)
  close(10)
  if (write_wlen) then
     open(unit=11,file='short_stream/wlen.dat')
     write(11,*)
     close(11)
  end if
  ! Do actual writing to file
  open(unit=10,file=trim(adjustl(filename)), &
       form ='unformatted', ACCESS = 'stream')
   if (write_wlen) then
     open(unit=11,file='short_stream/wlen.dat', &
       form ='unformatted', ACCESS = 'stream')
  end if
  !write(*,*) lamb_min, lamb_max, lamb(1), lamb(lamb_len)
  do i_lamb = 1, lamb_len-1
     ! Test for wlen with index+1, beause we want to be including the
     ! minimum wlen in our grid.
     if (lamb(i_lamb+1) > lamb_min) then
        write(10) opacity(i_lamb)
        if (write_wlen) then
           write(11) lamb(i_lamb)
        end if
     end if
     ! Same holds for upper boundary: we want to be inclusive!
     if (lamb(i_lamb) > lamb_max) then
        write(10) opacity(i_lamb)
        if (write_wlen) then
           write(11) lamb(i_lamb)
        end if
        EXIT
     end if
  end do
  write(*,*) i_lamb
  close(10)
  if (write_wlen) then
     close(11)
  end if

end subroutine write_opa

subroutine get_opa_dim(lamb_len, filename)

  implicit none

  CHARACTER*150                 :: filename
  INTEGER                       :: lamb_len

  lamb_len = 0
  open(unit=10,file=trim(adjustl(filename)))
  do while (1>0)
     read(10,*,end=10)
     lamb_len = lamb_len + 1
  end do
  10 close(10)

end subroutine get_opa_dim

subroutine read_wlen_opa(lamb, opacity, lamb_len, filename)

  implicit none

  CHARACTER*150                 :: filename
  INTEGER                       :: lamb_len, i_lamb
  DOUBLE PRECISION              :: lamb(lamb_len), opacity(lamb_len)

  open(unit=10,file=trim(adjustl(filename)))
  do i_lamb = 1, lamb_len
     read(10,*) lamb(i_lamb), opacity(i_lamb)
  end do
  close(10)

end subroutine read_wlen_opa

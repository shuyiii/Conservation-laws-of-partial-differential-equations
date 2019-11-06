!  --------------------------------------------------
!  Silverfrost FTN95 for Microsoft Visual Studio
!  Free Format FTN95 Source File
!  --------------------------------------------------
module constants  
implicit none
real*8, dimension(24) :: b1,b2,b3,b4,b5!bias
real*8, dimension (201) :: b6!bias of dense layer
real*8, dimension(2,1,24) :: c1!convolutional weights
real*8, dimension(2,24,24) :: c2,c3,c4,c5!convolutional weights
real*8, dimension(2376,201) :: c6!dense weights

contains
  subroutine loaddata
        open (unit = 10, file = "./c1.txt")
        read(10,*) c1
        open (unit = 11, file = "./c2.txt")
        read(11,*) c2
        open (unit = 12, file = "./c3.txt")
        read(12,*) c3
        open (unit = 13, file = "./c4.txt")
        read(13,*) c4
        open (unit = 14, file = "./c5.txt")
        read(14,*) c5
        open (unit = 15, file = "./c6.txt")
        read(15,*) c6
        open (unit = 16, file = "./b1.txt")
        read(16,*) b1
        open (unit = 17, file = "./b2.txt")
        read(17,*) b2
        open (unit = 18, file = "./b3.txt")
        read(18,*) b3
        open (unit = 19, file = "./b4.txt")
        read(19,*) b4
        open (unit = 20, file = "./b5.txt")
        read(20,*) b5
        open (unit = 21, file = "./b6.txt")
        read(21,*) b6
  end subroutine loaddata
end module constants


module functions
use constants
implicit none
contains
    subroutine norm(dat,N)!N is the length of data, need to normalize before prediction
    implicit none
    real*8,dimension(:) :: dat
    integer :: N
    real*8 :: sd,mean
    mean = SUM(dat(1:N)) / (N*1.0) 
    sd = SQRT (SUM((dat(1:N)-mean)**2) / (N*1.0))
    dat=(dat-mean)/sd 
    end subroutine norm

    ! this function give predictions 
    function oneDprediction(dat,len,step)   
    implicit none       
       integer ::len,step,s
       real*8,dimension(len) :: dat  
       real*8,dimension(202) :: dattemp
       real*8,dimension(201) :: outputtemp      
       real*8,dimension(len-1) :: oneDprediction
   
       oneDprediction=0d0
       if (len<202) then
       dattemp(1:(202-len)/2)=dat(1)
       dattemp((202-len)/2+1:(202-len)/2+len)=dat
       dattemp((202-len)/2+len+1:202)=dat(len)
       outputtemp=CNN(dattemp)
       oneDprediction=outputtemp((202-len)/2+1:(202-len)/2+len-1)
       else
       s=1
       do while(s+200.LE.len-1)
          oneDprediction(s:s+200)=MAX(oneDprediction(s:s+200),CNN(dat(s:s+201)))
          s=s+step
       end do
       oneDprediction(len-201:len-1)=MAX(oneDprediction(len-201:len-1),CNN(dat(len-201:len)))
       end if
    end function oneDprediction

    function CNN(input)!CNN model 
       implicit none 
       real*8,dimension(202) :: input
       real*8,dimension(202,1) :: input1
       real*8,dimension(201,24) :: input2
       real*8,dimension(200,24) :: input3
       real*8,dimension(199,24) :: input4
       real*8,dimension(198,24) :: input5
       real*8,dimension(99,24) :: input6temp
       real*8,dimension(2376) :: input6
       real*8,dimension(201) :: CNN
       
       input1=reshape(input,[202,1])
       call conv(input1,c1,b1,1,input2)
       call conv(input2,c2,b2,1,input3)
       call conv(input3,c3,b3,1,input4)
       call conv(input4,c4,b4,1,input5)
       call conv(input5,c5,b5,2,input6temp)
       input6=reshape(transpose(input6temp),[2376])!transpose becase of different reshape order of python and fortran
       call dense(input6,c6,b6,CNN)
    end function CNN

    subroutine conv(input,c,b,s,output)!convolutional layer only for 1d case
        real*8,dimension(:,:,:) :: c
        real*8,dimension(:,:) :: input,output
        real*8,dimension(:) :: b
        integer,dimension(2) :: a1
        integer,dimension(3) :: a2
        integer,dimension(1) ::m
        integer :: i,n,s
        real*8, dimension (:), allocatable :: t1,t2   

        a1=shape(input)
        a2=shape(c)
        m=shape(b)
        allocate (t1(a2(1)*a2(2)))
        allocate (t2(a2(1)*a2(2))) 
        do n=1,(a1(1)-a2(1))/s+1
              do i=1,m(1)
                 t1=reshape(input((n-1)*s+1:(n-1)*s+a2(1),:),[a2(1)*a2(2)])
                 t2=reshape(c(:,:,i),[a2(1)*a2(2)])
                 output(n,i)=MAX(dot_product(t1,t2)+b(i),0d0)!use relu in this convolutional layer, should be changed if use other activation functions
              end do
        end do
        deallocate(t1)
        deallocate(t2)
    end subroutine conv   

    subroutine dense(input,c,b,output)!dense layer
        real*8,dimension(:) :: input,output,b
        real*8,dimension(:,:) :: c
        integer :: n
        integer,dimension(1) :: m
        m=shape(output)
        do n=1,m(1)
           output(n)=dot_product(input,c(:,n))+b(n)
        end do
    end subroutine dense
end module functions



PROGRAM main
use constants
use functions
IMPLICIT NONE

real*8,dimension(100) :: data1
real*8,dimension(800) :: data2
real*8,dimension(99) ::  results1
real*8,dimension(799) :: results2

open (unit = 22, file = "./u_050_0000_n1p1_T0d4.txt")
read(22,*) data1
open (unit = 23, file = "./u_400_0000_n1p1_T0d4.txt")
read(23,*) data2

call loaddata

results1 = 0d0
results2 = 0d0

call norm(data1,100)!100 is the length of data
call norm(data2,800)!800 is the length of data

results1=oneDprediction(data1,100,0)!0 is the sliding window size (less then 202, no need to move)
results2=oneDprediction(data2,800,100)!100 is the sliding window size

write(*,*) results1
write(*,*) results2

PAUSE
END PROGRAM main
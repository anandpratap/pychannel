subroutine diff(ny, y, u, up)
  implicit none
  integer :: ny
  real(kind=8), dimension(ny) :: y, u, up

  up(2:ny-1) = (u(3:ny) - u(1:ny-2))/(y(3:ny) - y(1:ny-2))
  up(1) = (u(2) - u(1))/(y(2) - y(1))
  up(ny) = (u(ny) - u(ny-1))/(y(ny) - y(ny-1))
end subroutine diff

subroutine diff2(ny, y, u, upp)
  implicit none
  integer :: ny
  real(kind=8), dimension(ny) :: y, u, upp
  real(kind=8), dimension(ny) :: y_eta, u_eta2, y_eta2, uy
  call diff(ny, y, u, uy)
  
  y_eta2(2:ny-1) = (y(3:ny) - 2*y(2:ny-1) + y(1:ny-2))
  u_eta2(2:ny-1) = (u(3:ny) - 2*u(2:ny-1) + u(1:ny-2))
  y_eta(2:ny-1) = (y(3:ny) - y(1:ny-2))/2.0

  y_eta2(1) = (y(1) - 2*y(2) + y(3))
  u_eta2(1) = (u(1) - 2*u(2) + u(3))
  y_eta(1) = (y(2) - y(1))

  y_eta2(ny) = (y(ny) - 2*y(ny-1) + y(ny-2))
  u_eta2(ny) = (u(ny) - 2*u(ny-1) + u(ny-2))
  y_eta(ny) = (y(ny) - y(ny-1))
  upp = -(uy*y_eta2 - u_eta2)/(y_eta**2  + 1.e-20)
end subroutine diff2

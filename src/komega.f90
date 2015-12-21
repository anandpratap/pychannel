subroutine get_var(ny, q, u, k, omega)
  implicit none
  integer :: ny
  real(kind=8), dimension(ny*3) :: q
  real(kind=8), dimension(ny) :: u, k, omega
  integer :: i
  do i=1, ny
     u(i) = q(3*(i-1) + 1)
     k(i) = q(3*(i-1) + 2)
     omega(i) = q(3*(i-1) + 3)
  end do
end subroutine get_var
   
subroutine calc_residual(ny, y, q, R, dp, beta)
  implicit none
  integer :: ny, n
  real(kind=8), dimension(ny) :: y, u, k, omega, beta
!  real(kind=8), allocatable :: u(:), k(:), omega(:)
  real(kind=8), dimension(ny*3) :: q
  real(kind=8), dimension(ny*3), intent(out) :: R
  real(kind=8) :: dp
!  allocate(u(ny),k(ny),omega(ny))
  n = 3*ny
  call get_var(ny, q, u, k, omega)
  call calc_momentum_residual(ny, y, u, k, omega, R(1:n:3), dp)
  call calc_k_residual(ny, y, u, k, omega, R(2:n:3))
  call calc_omega_residual(ny, y, u, k, omega, R(3:n:3), beta)
!  deallocate(u,k,omega)
end subroutine calc_residual

subroutine calc_momentum_residual(ny, y, u, k, omega, R, dp)
  implicit none
  integer ny
  real(kind=8), dimension(ny) :: u, k, omega, R, y
  real(kind=8), dimension(ny) :: uy, uyy, nut, nuty
  real(kind=8) :: nu, rho, dp

  
  nu = 1e-4
  rho = 1.0

  call diff(ny, y, u, uy)
  call diff2(ny, y, u, uyy)
  nut = k/(omega + 1e-16)
  call diff(ny, y, nut, nuty)
  R = nu*uyy - dp/rho + nut*uyy + nuty*uy
  R(1) = -u(1)
  R(ny) = (1.5*u(ny) - 2*u(ny-1) + 0.5*u(ny-2))/(y(ny) - y(ny-1))
end subroutine calc_momentum_residual

subroutine calc_k_residual(ny, y, u, k, omega, R)
  implicit none
  integer ny
  real(kind=8), dimension(ny) :: u, k, omega, R, y
  real(kind=8), dimension(ny) :: uy, ky, kyy, omegay
  real(kind=8) :: beta_s, nu, sigma_k
  nu = 1e-4
  beta_s = 0.09
  sigma_k = 0.6

  call diff(ny, y, u, uy)
  call diff(ny, y, k, ky)
  call diff2(ny, y, k, kyy)
  call diff(ny, y, omega, omegay)
  

  R = k/omega*uy**2 - beta_s*k*omega + nu*kyy + sigma_k*(kyy*k/omega + ky*(ky/omega - k/omega**2*omegay))

  R(1) = -k(1)
  R(ny) = (1.5*k(ny) - 2*k(ny-1) + 0.5*k(ny-2))/(y(ny) - y(ny-1))
end subroutine calc_k_residual

subroutine calc_omega_residual(ny, y, u, k, omega, R, beta)
  implicit none
  integer ny
  real(kind=8), dimension(ny) :: u, k, omega, R, y, beta
  real(kind=8), dimension(ny) :: uy, ky, omegay, omegayy
  real(kind=8) :: gamma_w, beta_0, nu, sigma_w
  nu = 1e-4
  gamma_w = 13.0/25.0
  beta_0 = 0.0708
  sigma_w = 0.5

  call diff(ny, y, u, uy)
  call diff(ny, y, k, ky)
  call diff(ny, y, omega, omegay)
  call diff2(ny, y, omega, omegayy)


  R = beta*gamma_w*uy**2 - beta_0*omega**2 + nu*omegayy + sigma_w*(omegay*(ky/omega - k*omegay/omega**2) + k/omega*omegayy)
  R(1) = -(omega(1) - 5000000.0*nu/0.005**2)
  R(ny) = (1.5*omega(ny) - 2*omega(ny-1) + 0.5*omega(ny-2))/(y(ny) - y(ny-1))
end subroutine calc_omega_residual


! program main
!   implicit none
!   integer :: n
!   real(kind=8), allocatable::y(:), q(:), R(:)
!   real(kind=8) :: dp

!   n = 41
!   dp = -001.0
!   allocate(y(n), q(3*n), R(3*n))
!   call calc_residual(n, y, q, R, dp)
!   deallocate(y, q, R)

! end program main


subroutine calc_jacobian(ny, y, q, jac, dp, beta)
  implicit none
  integer :: ny, n
  real(kind=8), dimension(ny) :: y, u, k, omega, beta, betad
!  real(kind=8), allocatable :: u(:), k(:), omega(:)
  real(kind=8), dimension(ny*3) :: q, qd, r, rd
  real(kind=8), dimension(ny*3, ny*3), intent(out) :: jac
  real(kind=8) :: dp
  integer :: i
  n = ny*3
  rd(:) = 0.0
  qd(:) = 0.0
  do i=1,n
     betad(:) = 0.0
     qd(:) = 0.0
     rd(:) = 0.0
     qd(i) = 1.0
     call calc_residual_bq(ny, y, q, qd, r, rd, dp, beta, betad)
     jac(i,:) = rd(:)
  end do
end subroutine calc_jacobian

subroutine calc_delR_delbeta(ny, y, q, jac, dp, beta)
  implicit none
  integer :: ny, n
  real(kind=8), dimension(ny) :: y, u, k, omega, beta, betad
!  real(kind=8), allocatable :: u(:), k(:), omega(:)
  real(kind=8), dimension(ny*3) :: q, qd, r, rd
  real(kind=8), dimension(ny*3, ny), intent(out) :: jac
  real(kind=8) :: dp
  integer :: i
  n = ny*3
  rd(:) = 0.0
  qd(:) = 0.0
  do i=1,ny
     betad(:) = 0.0
     qd(:) = 0.0
     rd(:) = 0.0
     betad(i) = 1.0
     call calc_residual_bq(ny, y, q, qd, r, rd, dp, beta, betad)
     jac(:,i) = rd(:)
  end do
end subroutine calc_delR_delbeta


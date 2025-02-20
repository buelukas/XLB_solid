import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.helper import create_nse_fields, initialize_eq, check_bc_overlaps
from xlb.operator.boundary_masker import IndicesBoundaryMasker
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import HalfwayBounceBackBC, EquilibriumBC
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_fields_vtk, save_image
from xlb.grid import grid_factory
import xlb.velocity_set
import warp as wp
from warp import sin, cos, pi
import jax.numpy as jnp
import numpy as np
from typing import Any

if __name__ == "__main__":
    # Running the simulation
    total_time = 1.0
    n_steps = 100
    pp = False
    n_pp = 100
    pp_freq = int(np.ceil(n_steps/n_pp))
    domain_size = 1.0
    grid_size = 3
    grid_shape = (grid_size, grid_size)
    backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D2Q4(precision_policy=precision_policy, backend=backend)
    delta_x = domain_size/grid_size
    delta_t = total_time/n_steps
    c = delta_x/delta_t
    c_k = 1.1**0.5
    c_m = 0.4**0.5
    stability_factor = 2.0*np.sqrt(c_k**2+c_m**2.0)/c
    grid = grid_factory(grid_shape, compute_backend=backend)
    vector_size = 5
    cardinality = vector_size * velocity_set.q
    f = grid.create_field(cardinality=cardinality, dtype=precision_policy.store_precision)
    fstar = grid.create_field(cardinality=cardinality, dtype=precision_policy.store_precision)
    feq = grid.create_field(cardinality=cardinality, dtype=precision_policy.store_precision)
    U_num = grid.create_field(cardinality=vector_size, dtype=precision_policy.store_precision)
    B_lb = grid.create_field(cardinality=vector_size, dtype=precision_policy.store_precision)
    _vector_vec = wp.vec(vector_size, dtype=precision_policy.compute_precision.wp_dtype)
    _vector_mat = wp.types.matrix(shape=(vector_size, velocity_set.q), dtype=precision_policy.compute_precision.wp_dtype)

    #function to retrieve the analytical solution of U
    @wp.func
    def U(x: wp.float32, y: wp.float32, t: wp.float32):
        U = _vector_vec()
        U[0] = 1.6*pi*sin(pi*(-1.6*t + 2.0*y))*sin(pi*(-1.2*t + 4.0*x))*sin(pi*(4.0*t - 0.4)) + 4.0*pi*sin(pi*(-1.2*t + 4.0*x))*cos(pi*(-1.6*t + 2.0*y))*cos(pi*(4.0*t - 0.4)) - 1.2*pi*sin(pi*(4.0*t - 0.4))*cos(pi*(-1.6*t + 2.0*y))*cos(pi*(-1.2*t + 4.0*x))
        U[1] = 2.8*pi*sin(pi*(-2.8*t + 4.0*x))*sin(pi*(-0.2*t + 2.0*y))*cos(pi*(4.0*t + 1.6)) - 4.0*pi*sin(pi*(-0.2*t + 2.0*y))*sin(pi*(4.0*t + 1.6))*cos(pi*(-2.8*t + 4.0*x)) - 0.2*pi*cos(pi*(-2.8*t + 4.0*x))*cos(pi*(-0.2*t + 2.0*y))*cos(pi*(4.0*t + 1.6))
        U[2] = -c_k*(4.0*pi*sin(pi*(4.0*t - 0.4))*cos(pi*(-1.6*t + 2.0*y))*cos(pi*(-1.2*t + 4.0*x))+2.0*pi*cos(pi*(-2.8*t + 4.0*x))*cos(pi*(-0.2*t + 2.0*y))*cos(pi*(4.0*t + 1.6)))
        U[3] = -c_m*(4.0*pi*sin(pi*(4.0*t - 0.4))*cos(pi*(-1.6*t + 2.0*y))*cos(pi*(-1.2*t + 4.0*x))-2.0*pi*cos(pi*(-2.8*t + 4.0*x))*cos(pi*(-0.2*t + 2.0*y))*cos(pi*(4.0*t + 1.6)))
        U[4] = -c_m*(-2.0*pi*sin(pi*(-1.6*t + 2.0*y))*sin(pi*(-1.2*t + 4.0*x))*sin(pi*(4.0*t - 0.4))-4.0*pi*sin(pi*(-2.8*t + 4.0*x))*sin(pi*(-0.2*t + 2.0*y))*cos(pi*(4.0*t + 1.6)))
        return U
    
    #function to retrieve the analytical solution of B
    @wp.func
    def B(x: wp.float32, y: wp.float32, t: wp.float32):
        B = _vector_vec()
        B[0] = -c_k**2.0*(-8.0*pi**2.0*sin(pi*(-2.8*t + 4.0*x))*cos(pi*(-0.2*t + 2.0*y))*cos(pi*(4.0*t + 1.6)) - 16.0*pi**2.0*sin(pi*(-1.2*t + 4.0*x))*sin(pi*(4.0*t - 0.4))*cos(pi*(-1.6*t + 2.0*y))) - c_m**2.0*(-8.0*pi**2.0*sin(pi*(-2.8*t + 4.0*x))*cos(pi*(-0.2*t + 2.0*y))*cos(pi*(4.0*t + 1.6)) - 4.0*pi**2.0*sin(pi*(-1.2*t + 4.0*x))*sin(pi*(4.0*t - 0.4))*cos(pi*(-1.6*t + 2.0*y))) - c_m**2.0*(8.0*pi**2.0*sin(pi*(-2.8*t + 4.0*x))*cos(pi*(-0.2*t + 2.0*y))*cos(pi*(4.0*t + 1.6)) - 16.0*pi**2.0*sin(pi*(-1.2*t + 4.0*x))*sin(pi*(4.0*t - 0.4))*cos(pi*(-1.6*t + 2.0*y))) + 12.8*pi**2.0*sin(pi*(-1.6*t + 2.0*y))*sin(pi*(-1.2*t + 4.0*x))*cos(pi*(4.0*t - 0.4)) - 3.84*pi**2.0*sin(pi*(-1.6*t + 2.0*y))*sin(pi*(4.0*t - 0.4))*cos(pi*(-1.2*t + 4.0*x)) - 20.0*pi**2.0*sin(pi*(-1.2*t + 4.0*x))*sin(pi*(4.0*t - 0.4))*cos(pi*(-1.6*t + 2.0*y)) - 9.6*pi**2.0*cos(pi*(-1.6*t + 2.0*y))*cos(pi*(-1.2*t + 4.0*x))*cos(pi*(4.0*t - 0.4))
        B[1] = -c_k**2.0*(-8.0*pi**2.0*sin(pi*(-1.6*t + 2.0*y))*sin(pi*(4.0*t - 0.4))*cos(pi*(-1.2*t + 4.0*x)) - 4.0*pi**2.0*sin(pi*(-0.2*t + 2.0*y))*cos(pi*(-2.8*t + 4.0*x))*cos(pi*(4.0*t + 1.6))) - c_m**2.0*(-8.0*pi**2.0*sin(pi*(-1.6*t + 2.0*y))*sin(pi*(4.0*t - 0.4))*cos(pi*(-1.2*t + 4.0*x)) - 16.0*pi**2.0*sin(pi*(-0.2*t + 2.0*y))*cos(pi*(-2.8*t + 4.0*x))*cos(pi*(4.0*t + 1.6))) + c_m**2.0*(-8.0*pi**2.0*sin(pi*(-1.6*t + 2.0*y))*sin(pi*(4.0*t - 0.4))*cos(pi*(-1.2*t + 4.0*x)) + 4.0*pi**2.0*sin(pi*(-0.2*t + 2.0*y))*cos(pi*(-2.8*t + 4.0*x))*cos(pi*(4.0*t + 1.6))) - 22.4*pi**2.0*sin(pi*(-2.8*t + 4.0*x))*sin(pi*(-0.2*t + 2.0*y))*sin(pi*(4.0*t + 1.6)) - 1.12*pi**2.0*sin(pi*(-2.8*t + 4.0*x))*cos(pi*(-0.2*t + 2.0*y))*cos(pi*(4.0*t + 1.6)) - 23.88*pi**2.0*sin(pi*(-0.2*t + 2.0*y))*cos(pi*(-2.8*t + 4.0*x))*cos(pi*(4.0*t + 1.6)) + 1.6*pi**2.0*sin(pi*(4.0*t + 1.6))*cos(pi*(-2.8*t + 4.0*x))*cos(pi*(-0.2*t + 2.0*y))
        B[2] = 0.0
        B[3] = 0.0
        B[4] = 0.0
        return B

    #function to retrieve the analytical solution of dUdx
    @wp.func
    def dUdx(x: wp.float32, y: wp.float32, t: wp.float32):
        dUdx = _vector_vec()
        dUdx[0] = 6.4*pi**2.0*sin(pi*(-1.6*t + 2.0*y))*sin(pi*(4.0*t - 0.4))*cos(pi*(-1.2*t + 4.0*x)) + 4.8*pi**2.0*sin(pi*(-1.2*t + 4.0*x))*sin(pi*(4.0*t - 0.4))*cos(pi*(-1.6*t + 2.0*y)) + 16.0*pi**2.0*cos(pi*(-1.6*t + 2.0*y))*cos(pi*(-1.2*t + 4.0*x))*cos(pi*(4.0*t - 0.4))
        dUdx[1] = 16.0*pi**2.0*sin(pi*(-2.8*t + 4.0*x))*sin(pi*(-0.2*t + 2.0*y))*sin(pi*(4.0*t + 1.6)) + 0.8*pi**2.0*sin(pi*(-2.8*t + 4.0*x))*cos(pi*(-0.2*t + 2.0*y))*cos(pi*(4.0*t + 1.6)) + 11.2*pi**2.0*sin(pi*(-0.2*t + 2.0*y))*cos(pi*(-2.8*t + 4.0*x))*cos(pi*(4.0*t + 1.6))
        dUdx[2] = -c_k*(-8.0*pi**2.0*sin(pi*(-2.8*t + 4.0*x))*cos(pi*(-0.2*t + 2.0*y))*cos(pi*(4.0*t + 1.6)) - 16.0*pi**2.0*sin(pi*(-1.2*t + 4.0*x))*sin(pi*(4.0*t - 0.4))*cos(pi*(-1.6*t + 2.0*y)))
        dUdx[3] = -c_m*(8.0*pi**2.0*sin(pi*(-2.8*t + 4.0*x))*cos(pi*(-0.2*t + 2.0*y))*cos(pi*(4.0*t + 1.6)) - 16.0*pi**2.0*sin(pi*(-1.2*t + 4.0*x))*sin(pi*(4.0*t - 0.4))*cos(pi*(-1.6*t + 2.0*y))) 
        dUdx[4] = -c_m*(-8.0*pi**2.0*sin(pi*(-1.6*t + 2.0*y))*sin(pi*(4.0*t - 0.4))*cos(pi*(-1.2*t + 4.0*x)) - 16.0*pi**2.0*sin(pi*(-0.2*t + 2.0*y))*cos(pi*(-2.8*t + 4.0*x))*cos(pi*(4.0*t + 1.6)))
        return dUdx
    
    #function to retrieve the analytical solution of dUdy    
    @wp.func
    def dUdy(x: wp.float32, y: wp.float32, t: wp.float32):
        dUdy = _vector_vec()
        dUdy[0] = -8.0*pi**2.0*sin(pi*(-1.6*t + 2.0*y))*sin(pi*(-1.2*t + 4.0*x))*cos(pi*(4.0*t - 0.4)) + 2.4*pi**2.0*sin(pi*(-1.6*t + 2.0*y))*sin(pi*(4.0*t - 0.4))*cos(pi*(-1.2*t + 4.0*x)) + 3.2*pi**2.0*sin(pi*(-1.2*t + 4.0*x))*sin(pi*(4.0*t - 0.4))*cos(pi*(-1.6*t + 2.0*y))
        dUdy[1] = 5.6*pi**2.0*sin(pi*(-2.8*t + 4.0*x))*cos(pi*(-0.2*t + 2.0*y))*cos(pi*(4.0*t + 1.6)) + 0.4*pi**2.0*sin(pi*(-0.2*t + 2.0*y))*cos(pi*(-2.8*t + 4.0*x))*cos(pi*(4.0*t + 1.6)) - 8.0*pi**2.0*sin(pi*(4.0*t + 1.6))*cos(pi*(-2.8*t + 4.0*x))*cos(pi*(-0.2*t + 2.0*y))
        dUdy[2] = -c_k*(-8.0*pi**2.0*sin(pi*(-1.6*t + 2.0*y))*sin(pi*(4.0*t - 0.4))*cos(pi*(-1.2*t + 4.0*x)) - 4.0*pi**2.0*sin(pi*(-0.2*t + 2.0*y))*cos(pi*(-2.8*t + 4.0*x))*cos(pi*(4.0*t + 1.6)))
        dUdy[3] = -c_m*(-8.0*pi**2.0*sin(pi*(-1.6*t + 2.0*y))*sin(pi*(4.0*t - 0.4))*cos(pi*(-1.2*t + 4.0*x)) + 4.0*pi**2.0*sin(pi*(-0.2*t + 2.0*y))*cos(pi*(-2.8*t + 4.0*x))*cos(pi*(4.0*t + 1.6)))
        dUdy[4] = -c_m*(-8.0*pi**2.0*sin(pi*(-2.8*t + 4.0*x))*cos(pi*(-0.2*t + 2.0*y))*cos(pi*(4.0*t + 1.6)) - 4.0*pi**2.0*sin(pi*(-1.2*t + 4.0*x))*sin(pi*(4.0*t - 0.4))*cos(pi*(-1.6*t + 2.0*y)))
        return dUdy
    
    @wp.func
    def read_pop_functional(f: Any, index: Any):
        _f = _vector_mat()
        for ij in range(velocity_set.q):
            for s in range(vector_size):
                _f[s, ij] = f[s + vector_size * ij, index[0], index[1], index[2]]
        return _f

    @wp.func
    def write_pop_functional(f: Any, index: Any, _f: Any):
        for ij in range(velocity_set.q):
            for s in range(vector_size):
                f[s + vector_size * ij, index[0], index[1], index[2]] = precision_policy.store_precision.wp_dtype(_f[s, ij])

    @wp.func
    def read_U_num(U_num: Any, index: Any):
        _U = _vector_vec()
        for j in range(vector_size):
            _U[j] = U_num[j, index[0], index[1], index[2]]
        return _U

    @wp.func
    def write_U_num(U_num: Any, index: Any, _U: Any):
        for j in range(vector_size):
            U_num[j, index[0], index[1], index[2]] = precision_policy.store_precision.wp_dtype(_U[j])

    @wp.func
    def phi_x_lb(_U: Any):
        phi_x = _vector_vec()
        phi_x[0] = (c_k*_U[2]+c_m*_U[3])/c
        phi_x[1] = c_m*_U[4]/c
        phi_x[2] = c_k*_U[0]/c
        phi_x[3] = c_m*_U[0]/c
        phi_x[4] = c_m*_U[1]/c
        return phi_x

    @wp.func
    def phi_y_lb(_U: Any):
        phi_y = _vector_vec()
        phi_y[0] = c_m*_U[4]/c
        phi_y[1] = (c_k*_U[2]-c_m*_U[3])/c
        phi_y[2] = c_k*_U[1]/c
        phi_y[3] = -c_m*_U[1]/c
        phi_y[4] = c_m*_U[0]/c
        return phi_y

    #1st order IC
    @wp.kernel
    def initial_conditions(U_num: wp.array4d(dtype=Any), f: wp.array4d(dtype=Any)):
        i, j, k = wp.tid()
        index = wp.vec3i(i, j, k)
        x = (wp.float32(i)+0.5)*delta_x
        y = (wp.float32(j)+0.5)*delta_x
        t = 0.0
        #evaluate relevant properties from analytical solutions
        _U = U(x, y, t)
        _f = _vector_mat()
        _f0 = 0.25*(_U+2.0*phi_x_lb(_U))
        _f1 = 0.25*(_U+2.0*phi_y_lb(_U))
        _f2 = 0.25*(_U-2.0*phi_x_lb(_U))
        _f3 = 0.25*(_U-2.0*phi_y_lb(_U))
        for s in range(vector_size):
            _f[s,0] = _f0[s]
            _f[s,1] = _f1[s]
            _f[s,2] = _f2[s]
            _f[s,3] = _f3[s]
        write_pop_functional(f, index, _f)

    #2nd order IC with relatively high error due to precision issues
    @wp.kernel
    def initial_conditions_v2(U_num: wp.array4d(dtype=Any), f: wp.array4d(dtype=Any)):
        i, j, k = wp.tid()
        index = wp.vec3i(i, j, k)
        x = (wp.float32(i)+0.5)*delta_x
        y = (wp.float32(j)+0.5)*delta_x
        t = 0.0
        #evaluate relevant properties from analytical solutions
        _U = U(x, y, t)
        _B = B(x, y, t)*delta_t
        _dUdx = dUdx(x, y, t)*delta_x
        _dUdy = dUdy(x, y, t)*delta_x
        _f = _vector_mat()
        _f0 = 0.25*((_U+2.0*phi_x_lb(_U)) + 0.5*(-(_B+2.0*phi_x_lb(_B)) - _dUdx - phi_x_lb(_dUdx) + phi_y_lb(_dUdy) + 2.0*(phi_x_lb(phi_x_lb(_dUdx)+phi_y_lb(_dUdy)))))
        _f1 = 0.25*((_U+2.0*phi_y_lb(_U)) + 0.5*(-(_B+2.0*phi_y_lb(_B)) - _dUdy + phi_x_lb(_dUdx) - phi_y_lb(_dUdy) + 2.0*(phi_y_lb(phi_x_lb(_dUdx)+phi_y_lb(_dUdy)))))
        _f2 = 0.25*((_U-2.0*phi_x_lb(_U)) + 0.5*(-(_B-2.0*phi_x_lb(_B)) + _dUdx - phi_x_lb(_dUdx) + phi_y_lb(_dUdy) - 2.0*(phi_x_lb(phi_x_lb(_dUdx)+phi_y_lb(_dUdy)))))
        _f3 = 0.25*((_U-2.0*phi_y_lb(_U)) + 0.5*(-(_B-2.0*phi_y_lb(_B)) + _dUdy + phi_x_lb(_dUdx) - phi_y_lb(_dUdy) - 2.0*(phi_y_lb(phi_x_lb(_dUdx)+phi_y_lb(_dUdy)))))
        for s in range(vector_size):
            _f[s,0] = _f0[s]
            _f[s,1] = _f1[s]
            _f[s,2] = _f2[s]
            _f[s,3] = _f3[s]
        write_pop_functional(f, index, _f)

    @wp.kernel
    def collision_operator(U_num: wp.array4d(dtype=Any), f: wp.array4d(dtype=Any), fstar: wp.array4d(dtype=Any), feq: wp.array4d(dtype=Any), B_lb: wp.array4d(dtype=Any), n:wp.int32):
        # initialise necessary variables
        i, j, k = wp.tid()
        index = wp.vec3i(i, j, k)
        _U = _vector_vec()
        _B_lb = _vector_vec()
        _f = read_pop_functional(f, index)
        _feq = _vector_mat()
        _fstar = _vector_mat()
        # convert lattice units to actual coordinates
        x = (wp.float32(i)+0.5)*delta_x
        y = (wp.float32(j)+0.5)*delta_x
        t = wp.float32(n)*delta_t
        # evaluate forcing terms
        _B_lb = B(x, y, t)*delta_t
        write_U_num(B_lb, index, _B_lb)
        # compute _U(_f, bx, by) and store it globally
        for s in range(vector_size):
            _U[s] = _f[s,0]+_f[s,1]+_f[s,2]+_f[s,3]+0.5*_B_lb[s]
        write_U_num(U_num, index, _U)
        # compute phi_x(_U) and phi_y(_U)
        phi_x_lb = phi_x_lb(_U)
        phi_y_lb = phi_y_lb(_U)
        # compute _feq(_U, phi_x, phi_y) and store it globally
        for s in range(vector_size):
            _feq[s,0] = 0.25*(_U[s]) + 0.5*phi_x_lb[s]
            _feq[s,1] = 0.25*(_U[s]) + 0.5*phi_y_lb[s]
            _feq[s,2] = 0.25*(_U[s]) - 0.5*phi_x_lb[s]
            _feq[s,3] = 0.25*(_U[s]) - 0.5*phi_y_lb[s]
        write_pop_functional(feq, index, _feq)
        # compute _fstar(_U, phi_x, phi_y) and store it globally
        _fstar = 2.0*_feq-_f
        write_pop_functional(fstar, index, _fstar)

    @wp.kernel
    def streaming_operator(f: wp.array4d(dtype=Any), fstar: wp.array4d(dtype=Any)):
        i, j, k = wp.tid()
        index = wp.vec3i(i, j, k)
        i_plus = (i+1)%grid_size
        i_minus = (i-1+grid_size)%grid_size
        j_plus = (j+1)%grid_size
        j_minus = (j-1+grid_size)%grid_size
        _fstar = read_pop_functional(fstar, index)
        for s in range(vector_size):
            f[s + vector_size * 0, i_plus , j      , k] = precision_policy.store_precision.wp_dtype(_fstar[s,0])
            f[s + vector_size * 1, i      , j_plus , k] = precision_policy.store_precision.wp_dtype(_fstar[s,1])
            f[s + vector_size * 2, i_minus, j      , k] = precision_policy.store_precision.wp_dtype(_fstar[s,2])
            f[s + vector_size * 3, i      , j_minus, k] = precision_policy.store_precision.wp_dtype(_fstar[s,3])

if stability_factor<1:
    # wp.launch(initial_conditions, inputs=[U_num, f], dim=f.shape[1:])
    wp.launch(initial_conditions_v2, inputs=[U_num, f], dim=f.shape[1:])
    # wait for all kernels to finish computing
    wp.synchronize_device()
    for n in range(0, n_steps):
        # print("timestep =", n)
        # collision operator
        wp.launch(collision_operator, inputs=[U_num, f, fstar, feq, B_lb, n], dim=f.shape[1:])
        # wait for all kernels to finish computing
        wp.synchronize_device()
        # streaming operator
        wp.launch(streaming_operator, inputs=[f, fstar], dim=f.shape[1:])
        # wait for all kernels to finish computing
        wp.synchronize_device()
        # postprocessing
        if pp == True and n % pp_freq == 0:
            v_x = U_num.numpy()[0,:,:,0]
            v_y = U_num.numpy()[1,:,:,0]
            v_magnitude = np.sqrt((np.square(v_x)+np.square(v_y)))
            fields = {"v_x": v_x, "v_y": v_y, "v_magnitude": v_magnitude}
            save_fields_vtk(fields, timestep=n, prefix="elastodynamic_grid_test")
            save_image(fields["v_magnitude"], timestep=n, prefix="elastodynamic_grid_test")
    print("done")
else:
    print("unstable parameters: stability factor =", np.round(stability_factor, decimals=2), ", needs to be less than 1")

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
import jax.numpy as jnp
import numpy as np
from typing import Any

if __name__ == "__main__":
    # Running the simulation
    grid_size = 3
    grid_shape = (grid_size, grid_size)
    backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32

    velocity_set = xlb.velocity_set.D2Q4(precision_policy=precision_policy, backend=backend)
    omega = 1.0
    delta_x = 1.0
    delta_t = 1.0
    c = delta_x/delta_t
    c_k = 1.0
    c_m = 1.0

    grid = grid_factory(grid_shape, compute_backend=backend)
    vector_size = 5
    cardinality = vector_size * velocity_set.q
    f = grid.create_field(cardinality=cardinality, dtype=precision_policy.store_precision)
    feq = grid.create_field(cardinality=cardinality, dtype=precision_policy.store_precision)
    U_num = grid.create_field(cardinality=vector_size, dtype=precision_policy.store_precision)
    B = grid.create_field(cardinality=vector_size, dtype=precision_policy.store_precision)
    _vector_vec = wp.vec(vector_size, dtype=precision_policy.compute_precision.wp_dtype)
    _vector_mat = wp.types.matrix(shape=(vector_size, velocity_set.q), dtype=precision_policy.compute_precision.wp_dtype)

    @wp.func
    def read_pop_functional(f: Any, index: Any):
        _f = _vector_mat()
        for i in range(velocity_set.q):
            for j in range(vector_size):
                _f[i, j] = f[j + vector_size * i, index[0], index[1], index[2]]
        return _f

    @wp.func
    def write_pop_functional(f: Any, index: Any, _f: Any):
        for i in range(velocity_set.q):
            for j in range(vector_size):
                f[j + vector_size * i, index[0], index[1], index[2]] = precision_policy.store_precision.wp_dtype(_f[i, j])
    
    @wp.func
    def write_U_num(U_num: Any, index: Any, _U: Any):
        for j in range(vector_size):
            U_num[j, index[0], index[1], index[2]] = precision_policy.store_precision.wp_dtype(_U[j])
    
    @wp.func
    def phi_x(_U: Any, c_k: Any, c_m: Any):
        phi_x = _vector_vec()
        phi_x[0] = c_k*_U[2]+c_m*_U[3]
        phi_x[1] = c_m*_U[4]
        phi_x[2] = c_k*_U[0]
        phi_x[3] = c_m*_U[0]
        phi_x[4] = c_m*_U[1]
        return phi_x
    
    @wp.func
    def phi_y(_U: Any, c_k: Any, c_m: Any):
        phi_y = _vector_vec()
        phi_y[0] = c_m*_U[4]
        phi_y[1] = c_k*_U[2]-c_m*_U[3]
        phi_y[2] = c_k*_U[1]
        phi_y[3] = -c_m*_U[1]
        phi_y[4] = c_m*_U[0]
        return phi_y
    
    @wp.func
    def _feq(_U: Any, phi_x: Any, phi_y: Any):
        _feq = _vector_mat()
        for i in range(vector_size):
            for j in range(velocity_set.q):

                _feq[i,j]=(_U[i]+2.0/c*(velocity_set._c_float[0,j]*phi_x[i]+velocity_set._c_float[1,j]*phi_y[i]))/4.0


        return _feq

    @wp.kernel
    def init_kernel(f: wp.array4d(dtype=Any), U_num: wp.array4d(dtype=Any), B: wp.array4d(dtype=Any), feq: wp.array4d(dtype=Any)):
        i, j, k = wp.tid()
        index = wp.vec3i(i, j, k)
        _f = read_pop_functional(f, index)
        for i in range(vector_size):
            for j in range(velocity_set.q):
                _f[i, j] = precision_policy.compute_precision.wp_dtype(1.0)
        write_pop_functional(f, index, _f)

        _U = _vector_vec()
        for i in range(vector_size):
            for j in range(velocity_set.q):
                _U[i] = _U[i] + _f[i,j]
            _U[i] = _U[i] + B[i, index[0], index[1], index[2]]*delta_t/2.0
        write_U_num(U_num, index, _U)

        phi_x = phi_x(_U, c_k, c_m)
        phi_y = phi_y(_U, c_k, c_m)

        
        _feq = _feq(_U, phi_x, phi_y)
        write_pop_functional(feq, index, _feq)

wp.launch(init_kernel, inputs=[f,U_num,B,feq], dim=f.shape[1:])
print(f.numpy())
print(U_num.numpy())
print(feq.numpy())
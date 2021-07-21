/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
The class dealing with the RDIP potential.
------------------------------------------------------------------------------*/

#include "lj.cuh"
#include "utilities/error.cuh"
#define BLOCK_SIZE_FORCE 128

LJ::LJ(FILE* fid, int num_types)
{
  printf("Use the RDIP potential.\n");
  int count = fscanf(
    fid, "%f%f%f%f%f%f%f%f%f%f", &lj_para.A, &lj_para.B, &lj_para.C, &lj_para.D1, &lj_para.D2,
    &lj_para.z0, &lj_para.alpha, &lj_para.lambda1, &lj_para.lambda2, &lj_para.rc);
  PRINT_SCANF_ERROR(count, 10, "Reading error for RDIP potential.");
  lj_para.z02 = lj_para.z0 * lj_para.z0;
  lj_para.Az06 = lj_para.A * lj_para.z02 * lj_para.z02 * lj_para.z02;
  rc = lj_para.rc;
}

LJ::~LJ(void) {}

static __global__ void gpu_find_force(
  LJ_Para lj,
  const int number_of_particles,
  const int N1,
  const int N2,
  const Box box,
  const int* g_neighbor_number,
  const int* g_neighbor_list,
  const int* g_type,
  const int shift,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_potential)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index
  double s_fx = 0.0;                                   // force_x
  double s_fy = 0.0;                                   // force_y
  double s_fz = 0.0;                                   // force_z
  double s_pe = 0.0;                                   // potential energy
  double s_sxx = 0.0;                                  // virial_stress_xx
  double s_sxy = 0.0;                                  // virial_stress_xy
  double s_sxz = 0.0;                                  // virial_stress_xz
  double s_syx = 0.0;                                  // virial_stress_yx
  double s_syy = 0.0;                                  // virial_stress_yy
  double s_syz = 0.0;                                  // virial_stress_yz
  double s_szx = 0.0;                                  // virial_stress_zx
  double s_szy = 0.0;                                  // virial_stress_zy
  double s_szz = 0.0;                                  // virial_stress_zz

  if (n1 < N2) {
    int neighbor_number = g_neighbor_number[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int n2 = g_neighbor_list[n1 + number_of_particles * i1];

      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float x12 = float(x12double);
      float y12 = float(y12double);
      float z12 = float(z12double);

      float rhosq = x12 * x12 + y12 * y12;
      float d12sq = rhosq + z12 * z12;
      float d12 = sqrt(d12sq);
      float d12inv = 1 / d12;
      float d12inv2 = d12inv * d12inv;
      float d12inv4 = d12inv2 * d12inv2;
      float d12inv6 = d12inv4 * d12inv2;
      float d12inv8 = d12inv6 * d12inv2;

      float D_factor = lj.C * (1 + lj.D1 * rhosq + lj.D2 * rhosq * rhosq);
      float exp_alpha = lj.B * exp(-lj.alpha * (d12 - lj.z0));
      float exp_lambda = exp(-lj.lambda1 * rhosq - lj.lambda2 * (z12 * z12 - lj.z02));

      float tmp = -6.0f * lj.Az06 * d12inv8 - lj.alpha * exp_alpha * d12inv;
      float f12x = tmp * x12;
      float f12y = tmp * y12;
      float f12z = tmp * z12;
      tmp = 2 * exp_lambda * ((lj.D1 + 2 * lj.D2 * rhosq) * lj.C - lj.lambda1 * D_factor);
      f12x += tmp * x12;
      f12y += tmp * y12;
      tmp = -2 * lj.lambda2 * D_factor * exp_lambda;
      f12z += tmp * z12;

      s_pe += 0.5f * (lj.Az06 * d12inv6 + exp_alpha + D_factor * exp_lambda);
      s_fx += f12x;
      s_fy += f12y;
      s_fz += f12z;
      f12x *= 0.5f;
      f12y *= 0.5f;
      f12z *= 0.5f;
      s_sxx -= x12 * f12x;
      s_sxy -= x12 * f12y;
      s_sxz -= x12 * f12z;
      s_syx -= y12 * f12x;
      s_syy -= y12 * f12y;
      s_syz -= y12 * f12z;
      s_szx -= z12 * f12x;
      s_szy -= z12 * f12y;
      s_szz -= z12 * f12z;
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n1 + 0 * number_of_particles] += s_sxx;
    g_virial[n1 + 1 * number_of_particles] += s_syy;
    g_virial[n1 + 2 * number_of_particles] += s_szz;
    g_virial[n1 + 3 * number_of_particles] += s_sxy;
    g_virial[n1 + 4 * number_of_particles] += s_sxz;
    g_virial[n1 + 5 * number_of_particles] += s_syz;
    g_virial[n1 + 6 * number_of_particles] += s_syx;
    g_virial[n1 + 7 * number_of_particles] += s_szx;
    g_virial[n1 + 8 * number_of_particles] += s_szy;
    g_potential[n1] += s_pe;
  }
}

void LJ::compute(
  const int type_shift,
  const Box& box,
  const Neighbor& neighbor,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = type.size();
  int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;

  gpu_find_force<<<grid_size, BLOCK_SIZE_FORCE>>>(
    lj_para, number_of_atoms, N1, N2, box, neighbor.NN_local.data(), neighbor.NL_local.data(),
    type.data(), type_shift, position_per_atom.data(), position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2, force_per_atom.data(),
    force_per_atom.data() + number_of_atoms, force_per_atom.data() + 2 * number_of_atoms,
    virial_per_atom.data(), potential_per_atom.data());
  CUDA_CHECK_KERNEL
}

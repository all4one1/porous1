#pragma once
#include "cuda_runtime.h"


__device__  double dx1_center(unsigned int l, double *f) {
	return 0.5*(f[l + 1] - f[l - 1]) / dx[1];
}
__device__  double dy1_center(unsigned int l, double *f) {
	return 0.5*(f[l + offset] - f[l - offset]) / dy[1];
}
__device__  double dz1_center(unsigned int l, double *f) {
	return 0.5*(f[l + offset2] - f[l - offset2]) / dz[1];
}


__device__  double dx2_center(unsigned int l, double *f) {
	return (f[l + 1] - 2.0*f[l] + f[l - 1]) / dx[1] / dx[1];
}
__device__  double dy2_center(unsigned int l, double *f) {
	return (f[l + offset] - 2.0*f[l] + f[l - offset]) / dy[1] / dy[1];
}
__device__  double dz2_center(unsigned int l, double *f) {
	return (f[l + offset2] - 2.0*f[l] + f[l - offset2]) / dz[1] / dz[1];
}


__device__  double dx1_eq_0_forward(unsigned int l, double *f) {
	return (4.0*f[l + 1] - f[l + 2]) / 3.0;
}
__device__  double dx1_eq_0_back(unsigned int l, double *f) {
	return (4.0*f[l - 1] - f[l - 2]) / 3.0;
}

__device__  double dy1_eq_0_up(unsigned int l, double *f) {
	return (4.0*f[l + offset] - f[l + 2 * offset]) / 3.0;
}
__device__  double dy1_eq_0_down(unsigned int l, double *f) {
	return (4.0*f[l - offset] - f[l - 2 * offset]) / 3.0;
}

__device__  double dz1_eq_0_toDeep(unsigned int l, double *f) {
	return (4.0*f[l + offset2] - f[l + 2 * offset2]) / 3.0;
}
__device__  double dz1_eq_0_toUs(unsigned int l, double *f) {
	return (4.0*f[l - offset2] - f[l - 2 * offset2]) / 3.0;
}



__device__  double dx1_forward(unsigned int l, double *f) {
	return -0.5*(3.0*f[l] - 4.0*f[l + 1] + f[l + 2]) / dx[1];
}
__device__  double dx1_back(unsigned int l, double *f) {
	return  0.5*(3.0*f[l] - 4.0*f[l - 1] + f[l - 2]) / dx[1];
}
__device__  double dy1_up(unsigned int l, double *f) {
	return  -0.5*(3.0*f[l] - 4.0*f[l + offset] + f[l + 2 * offset]) / dy[1];
}
__device__  double dy1_down(unsigned int l, double *f) {
	return  0.5*(3.0*f[l] - 4.0*f[l - offset] + f[l - 2 * offset]) / dy[1];
}
__device__  double dz1_toDeep(unsigned int l, double *f) {
	return  -0.5*(3.0*f[l] - 4.0*f[l + offset2] + f[l + 2 * offset2]) / dz[1];
}
__device__  double dz1_toUs(unsigned int l, double *f) {
	return  0.5*(3.0*f[l] - 4.0*f[l - offset2] + f[l - 2 * offset2]) / dz[1];
}



__device__  double dx2_forward(unsigned int l, double *f) {
	return (2.0 * f[l] - 5.0 * f[l + 1] + 4.0 * f[l + 2] - f[l + 3]) / dx[1] / dx[1];
}
__device__  double dx2_back(unsigned int l, double *f) {
	return (2.0 * f[l] - 5.0 * f[l - 1] + 4.0 * f[l - 2] - f[l - 3]) / dx[1] / dx[1];
}
__device__  double dy2_up(unsigned int l, double *f) {
	return (2.0 * f[l] - 5.0 * f[l + offset] + 4.0 * f[l + 2 * offset] - f[l + 3 * offset]) / dy[1] / dy[1];
}
__device__  double dy2_down(unsigned int l, double *f) {
	return (2.0 * f[l] - 5.0 * f[l - offset] + 4.0 * f[l - 2 * offset] - f[l - 3 * offset]) / dy[1] / dy[1];
}
__device__  double dz2_toDeep(unsigned int l, double *f) {
	return (2.0 * f[l] - 5.0 * f[l + offset2] + 4.0 * f[l + 2 * offset2] - f[l + 3 * offset2]) / dz[1] / dz[1];
}
__device__  double dz2_toUs(unsigned int l, double *f) {
	return (2.0 * f[l] - 5.0 * f[l - offset2] + 4.0 * f[l - 2 * offset2] - f[l - 3 * offset2]) / dz[1] / dz[1];
}

__device__  double dx2_eq_0_forward(unsigned int l, double *f) {
	return (5.0 * f[l + 1] - 4.0 * f[l + 2] + f[l + 3]) * 0.5;
}
__device__  double dx2_eq_0_back(unsigned int l, double *f) {
	return (5.0 * f[l - 1] - 4.0 * f[l - 2] + f[l - 3]) * 0.5;
}
__device__  double dy2_eq_0_up(unsigned int l, double *f) {
	return (5.0 * f[l + offset] - 4.0 * f[l + 2 * offset] + f[l + 3 * offset]) * 0.5;
}
__device__  double dy2_eq_0_down(unsigned int l, double *f) {
	return (5.0 * f[l - offset] - 4.0 * f[l - 2 * offset] + f[l - 3 * offset]) * 0.5;
}
__device__  double dz2_eq_0_toDeep(unsigned int l, double *f) {
	return (5.0 * f[l + offset2] + 4.0 * f[l + 2 * offset2] - f[l + 3 * offset2]) * 0.5;
}
__device__  double dz2_eq_0_toUs(unsigned int l, double *f) {
	return (5.0 * f[l - offset2] + 4.0 * f[l - 2 * offset2] - f[l - 3 * offset2]) * 0.5;
}


__device__ double dx1(double *f, unsigned int l, unsigned int i) {
	if (i > 0 && i < nx_d) {
		return dx1_center(l, f);
	}
	else if (i == 0) {
		return dx1_forward(l, f);
	}
	else if (i == nx_d) {
		return dx1_back(l, f);
	}
	else {
		return 0;
	}
}
__device__ double dy1(double *f, unsigned int l, unsigned int j) {
	if (j > 0 && j < ny_d) {
		return dy1_center(l, f);
	}
	else if (j == 0) {
		return dy1_up(l, f);
	}
	else if (j == nx_d) {
		return dy1_down(l, f);
	}
	else {
		return 0;
	}
}
__device__ double dz1(double *f, unsigned int l, unsigned int k) {
	if (k > 0 && k < nz_d) {
		return dz1_center(l, f);
	}
	else if (k == 0) {
		return dz1_toDeep(l, f);
	}
	else if (k == nz_d) {
		return dz1_toUs(l, f);
	}
	else {
		return 0;
	}
}

__device__ double dx2(double *f, unsigned int l, unsigned int i) {
	if (i > 0 && i < nx_d) {
		return dx2_center(l, f);
	}
	else if (i == 0) {
		return dx2_forward(l, f);
	}
	else if (i == nx_d) {
		return dx2_back(l, f);
	}
	else {
		return 0;
	}
}
__device__ double dy2(double *f, unsigned int l, unsigned int j) {
	if (j > 0 && j < ny_d) {
		return dy2_center(l, f);
	}
	else if (j == 0) {
		return dy2_up(l, f);
	}
	else if (j == nx_d) {
		return dy2_down(l, f);
	}
	else {
		return 0;
	}
}
__device__ double dz2(double *f, unsigned int l, unsigned int k) {
	if (k > 0 && k < nz_d) {
		return dz2_center(l, f);
	}
	else if (k == 0) {
		return dz2_toDeep(l, f);
	}
	else if (k == nz_d) {
		return dz2_toUs(l, f);
	}
	else {
		return 0;
	}
}


__device__ double div(double *fx, double *fy, double *fz, unsigned int l, unsigned int i, unsigned int j, unsigned int k) {
	return dx1(fx, l, i) + dy1(fy, l, j) + dz1(fz, l, k);
}
__device__ double div(double *fx, double *fy, unsigned int l, unsigned int i, unsigned int j) {
	return dx1(fx, l, i) + dy1(fy, l, j);
}

__device__ double laplace(double *f, unsigned int l, unsigned int i, unsigned int j, unsigned int k) {
	return dx2(f, l, i) + dy2(f, l, j) + dz2(f, l, k);
}
__device__ double laplace(double *f, unsigned int l, unsigned int i, unsigned int j) {
	return dx2(f, l, i) + dy2(f, l, j);
}


__device__ double VgradF(double *vx, double *vy, double *F, unsigned int l, unsigned int i, unsigned int j) {

	double val = 0;

	if (vx[l] != 0) {
		val += vx[l] * dx1(F, l, i);
	}
	if (vy[l] != 0) {
		val += vy[l] * dy1(F, l, j);
	}
	return val;
}



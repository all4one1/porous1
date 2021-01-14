#define level2
#define NV_YES0
#define Stream
//#define BorderMesh

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <omp.h>
#include <iostream> 
#include <stdio.h> 
#include <string> 
#include <fstream> 
#include <iomanip> 
#include <sstream> 
#include <cstring> 
#include <cmath>
#include <algorithm> 
#include <cuda.h>
#include "Header.h"
#ifdef _WIN32
#include "windows.h"
#endif

#define Pi 3.1415926535897932384626433832795
#define pause system("pause");
#define timer timer2 = clock()/CLOCKS_PER_SEC; 	cout << "time (seconds)= " << (timer2 - timer1) << endl;
#define cudaCheckError() {cudaError_t e = cudaGetLastError();if (e != cudaSuccess) {printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(0);}}
#define _kernel_ << < gridD, blockD >> >
#define TEST cudaDeviceSynchronize(); write_test(); return 0;
#define VarLog(var) cout << #var << " " << var << endl;
#define check std::cout << "Line: " << __LINE__ << endl;
std::ofstream Log("LogFile.dat");

//CPU functions and variables 
using namespace std;
using std::cout;
int np;
double timer1, timer2;
double pi = 3.1415926535897932384626433832795;
double *p_d, *p0_d, *ux_d, *uy_d, *uz_d, *vx_d, *vy_d, *vz_d, *T_d, *T0_d, *C_d, *C0_d;  //_d - device (GPU) 
double *p_h, *p0_h, *ux_h, *uy_h, *uz_h, *vx_h, *vy_h, *vz_h, *T_h, *T0_h, *C_h, *C0_h;	//_h - host (CPU)
double *psiav_array, *psiav_d, eps_h, eps0;		 //  temporal variables
double *HX, *HY, *HZ, *XX, *YY, *ZZ, *DX, *DY, *DZ;
double hx, hy, hz, Lx, Ly, Lz, tau, tau_p, m, psiav, psiav0, eps; //parameters 
double Ek, Ek_old, Vmax, timeq, phi_check;
unsigned int nx, ny, nz, iter, niter, nout, nxout, nyout, off, off2, k, mx, my, mz, border, tt;					  //parameters
double Vxm, Vym, Vzm, pmax, Tm, Cmax, pmin, Tmin, Cmin, dCmax, dCmax0, dTmax, dTmax0;
double *test_d, *test_h, *test2_d, *test2_h;	  
double AMx, AMy, AMz, AMabs, eps_Heav_h;
double tau_ksi;
unsigned int size_l;
size_t size_b, thread_x_d, thread_y_d, thread_z_d;
double K, bettaT, bettaC, rho, chi, eta, grav, por, Dif, DifHeat, Temp0, Con0, T_cold, T_hot, deltaTemp, dT;
double rho_por, Cp, Cp_por, nu, D, Dt, Dt_, L, rhoCpF, rhoCpP, Ra, Le, porB, psi, St, St_, a, kappa_p;
int dimensionless = 1, read_recovery, grid_border;
double coefT, shiftT, coefC, shiftC, Ctotal, Ttotal, Ptotal, StopIncr, minimumTime;
unsigned long long int *Grid_p, *N_p;
unsigned int N_reduce, thread_p;
double *arr[10];
int copied;
int finish = 0, next_to = 0, first = 1;
ofstream Ra_tab, integrals;


//GPU functions and variables
__constant__ double hx_d, hy_d, hz_d, tau_d, Lx_d, Ly_d, Lz_d, Ra_d, Pr_d, tau_p_d, tau_ksi_d;
__constant__ unsigned int nx_d, ny_d, nz_d, n_d, offset, offset2, border_type;
__constant__ double eps0_d = 1e-5;
__constant__ double pi_d = 3.1415926535897932384626433832795;
__device__ double eps_d = 1.0; // psiav_d, psiav0_d = 0.0;
__device__ double dp;
__constant__ double K_d, bettaT_d, bettaC_d, rho_d, chi_d, eta_d, grav_d, por_d, Dif_d, DifHeat_d;
__constant__ double Temp0_d, Con0_d, T_cold_d, T_hot_d, deltaTemp_d, rho_por_d, Cp_d, Cp_por_d, nu_d;
__constant__ double D_d, Dt_d, rhoCpF_d, rhoCpP_d, Le_d, porB_d, psi_d, St_d, a_d;
__device__ double *dx, *dy, *dz;



double maxval(double* f, unsigned int n)
{
	double max = (f[0]);

	for (unsigned int i = 0; i < n; i++) {
		if ((f[i])>max)
		{
			max = (f[i]);
		}
	}
	return max;
}
double minval(double* f, unsigned int n)
{
	double min = (f[0]);

	for (unsigned int i = 0; i < n; i++) {
		if ((f[i])<min)
		{
			min = (f[i]);
		}
	}
	return min;
}
void maxval_index(double* f, unsigned int nx, unsigned int ny, unsigned int nz, unsigned int& mx, unsigned int& my, unsigned int& mz, double& max)
{
	max = f[0];
	int offset = nx + 1; int offset2 = (nx + 1) * (ny + 1);
	mx = 0; my = 0; mz = 0;


	for (unsigned int i = 0; i <= nx; i++) {
		for (unsigned int j = 0; j <= ny; j++) {
			for (unsigned int k = 0; k <= nz; k++) {
				if (abs(f[i + offset*j + offset2*k]) > abs(max))
				{
					max = (f[i + offset*j + offset2*k]);
					mx = i;
					my = j;
					mz = k;
				}
			}
		}
	}

}

double Int2d_point_simpson(double *f, int l, int i, int j) {
	double val = 0.0;


	if (i > 0 && i < nx  && j > 0 && j < ny) {
		val += 16.0 * f[l]
			+ 4.0 * f[l + 1]
			+ 4.0 * f[l - 1]
			+ 4.0 * f[l + off]
			+ 4.0 * f[l - off]
			+ f[l + 1 + off]
			+ f[l - 1 + off]
			+ f[l + 1 - off]
			+ f[l - 1 - off];
		val = val * hx * hy / 36.0;
	}
	else if ((i == 0 || i == nx) && (j > 0 && j < ny)) {
		val += 4.0 * f[l]
			+ f[l + off]
			+ f[l - off];
		val = 0.5 * val * hx * hy / 6.0;
	}
	else if ((j == 0 || j == ny) && (i > 0 && i < nx)) {
		val += 4.0 * f[l]
			+ f[l + 1]
			+ f[l - 1];
		val = 0.5 * val * hx * hy / 6.0;
	}
	else {
		val = 0.25 * f[l] * hx*hy;
	}
	return val;

}
double Int2d_point_simpson_temp(double *f, int l, int i, int j) {
	double val = 0.0;


	if (i > 0 && i < nx  && j > 0 && j < ny) {
		val += 16.0 * f[l]
			+ 4.0 * f[l + 1]
			+ 4.0 * f[l - 1]
			+ 4.0 * f[l + off]
			+ 4.0 * f[l - off]
			+ f[l + 1 + off]
			+ f[l - 1 + off]
			+ f[l + 1 - off]
			+ f[l - 1 - off];
		val = val * HX[i] * HY[j] / 36.0;
	}
	else if ((i == 0 || i == nx) && (j > 0 && j < ny)) {
		val += 4.0 * f[l]
			+ f[l + off]
			+ f[l - off];
		val = 0.5 * val * HX[0] * HY[j] / 6.0;
	}
	else if ((j == 0 || j == ny) && (i > 0 && i < nx)) {
		val += 4.0 * f[l]
			+ f[l + 1]
			+ f[l - 1];
		val = 0.5 * val * HX[i] * HY[0] / 6.0;
	}
	else {
		val = 0.25 * f[l] * HX[0] * HY[0];
	}
	return val;

}
double Int3d_point_simpson(double *f, int l, int i, int j, int k) {
	double val = 0.0;

	if (i > 0 && i < nx  && j > 0 && j < ny && k > 0 && k < nz) {
		val += 16.0 * f[l]
			+ 4.0 * f[l + 1]
			+ 4.0 * f[l - 1]
			+ 4.0 * f[l + off]
			+ 4.0 * f[l - off]
			+ f[l + 1 + off]
			+ f[l - 1 + off]
			+ f[l + 1 - off]
			+ f[l - 1 - off];
		val = val * hx * hy / 36.0;
	}
	else if ((i == 0 || i == nx) && (j > 0 && j < ny)) {
		val += 4.0 * f[l]
			+ f[l + off]
			+ f[l - off];
		val = 0.5 * val * hx * hy / 6.0;
	}
	else if ((j == 0 || j == ny) && (i > 0 && i < nx)) {
		val += 4.0 * f[l]
			+ f[l + 1]
			+ f[l - 1];
		val = 0.5 * val * hx * hy / 6.0;
	}
	else {
		val = 0.25 * f[l] * hx*hy;
	}
	return val;

}

double Integral(double *f) {
	double val = 0;

		int l;

		for (int i = 0; i <= nx; i++) {
			for (int j = 0; j <= ny; j++) {
				for (int k = 0; k <= nz; k++) {
					l = i + off*j + off2*k;
#ifdef BorderMesh
					val += Int2d_point_simpson_temp(f, l, i, j);
#else
					val += Int2d_point_simpson(f, l, i, j);
#endif // BorderMesh

					
					//val += Int2d_point_trap(f, l, i, j);
				}
			}
		}

	return val/(Lx*Ly);
	
}


#ifdef level3
void velocity() {
	double V = 0;
	Ek = 0.0; Vmax = 0.0;


	for (int qx = 1; qx <= nx - 1; qx++) {
		for (int qy = 1; qy <= ny - 1; qy++) {
			for (int qz = 1; qz <= nz - 1; qz++) {
				V =
					+vx_h[qx + off * qy + off2*qz] * vx_h[qx + off * qy + off2*qz]
					+ vy_h[qx + off * qy + off2*qz] * vy_h[qx + off * qy + off2*qz]
					+ vz_h[qx + off * qy + off2*qz] * vz_h[qx + off * qy + off2*qz];


				Ek += V;
				if (sqrt(V) > Vmax) Vmax = sqrt(V);
			}
		}
	}

	Ek = Ek / 2.0 * hx * hy * hz;


}
#endif
#ifdef level2
void velocity() {
	double V = 0;
	Ek = 0.0; Vmax = 0.0;


	for (int qx = 1; qx <= nx - 1; qx++) {
		for (int qy = 1; qy <= ny - 1; qy++) {
			V = +vx_h[qx + off * qy] * vx_h[qx + off * qy]
				+ vy_h[qx + off * qy] * vy_h[qx + off * qy];


			Ek += V;
			if (sqrt(V) > Vmax) Vmax = sqrt(V);
		}
	}

	Ek = Ek / 2.0 * hx * hy;

}

#endif 



double max_vertical_difference(double* f) {
	double max = 0, dif = 0;

	for (int i = 0; i <= nx; i++) {
		for (int k = 0; k <= nz; k++) {
			dif = f[i + off2*k] - f[i + off*ny + off2*k];
			dif = abs(dif);
			if (dif > max) max = dif;
		}
	}

	return max;
}

double Nu_y_down(double *f) {
	double N = 0;
	unsigned int offset = nx + 1;
	unsigned int offset2 = (ny + 1)*(nx + 1);
	double S = nx * hx * nz * hz;

	N += abs(f[0 + offset + offset2 * 0] - f[0 + offset2 * 0]) / hy;
	N += abs(f[0 + offset + offset2 * nz] - f[0 + offset2 * nz]) / hy;
	N += abs(f[nx + offset + offset2 * 0] - f[nx + offset2 * 0]) / hy;
	N += abs(f[nx + offset + offset2 * nz] - f[nx + offset2 * nz]) / hy;

	for (unsigned int i = 1; i <= nx - 1; i++)
		N += 2 * abs(f[i + offset + offset2 * 0] - f[i + offset2 * 0]) / hy;

	for (unsigned int i = 1; i <= nx - 1; i++)
		N += 2 * abs(f[i + offset + offset2 * nz] - f[i + offset2 * nz]) / hy;

	for (unsigned int k = 1; k <= nz - 1; k++)
		N += 2 * abs(f[0 + offset + offset2 * k] - f[0 + offset2 * k]) / hy;
	for (unsigned int k = 1; k <= nz - 1; k++)
		N += 2 * abs(f[nx + offset + offset2 * k] - f[nx + offset2 * k]) / hy;


	for (unsigned int i = 1; i <= nx - 1; i++) {
		for (unsigned int k = 1; k <= nz - 1; k++) {
			N += 4 * abs(f[i + offset + offset2 * k] - f[i + offset2 * k]) / hy;
		}
	}

	N = N  * (hx)* (hz) / 4 / S;
	return N;
}
double Nu_y_top(double *f) {
	double N = 0;
	unsigned int offset = nx + 1;
	unsigned int offset2 = (ny + 1)*(nx + 1);
	double S = nx * hx * nz * hz;


	N += abs(f[0 + offset*ny + offset2 * 0] - f[0 + offset*(ny - 1) + offset2 * 0]) / hy;
	N += abs(f[0 + offset*ny + offset2 * nz] - f[0 + offset*(ny - 1) + offset2 * nz]) / hy;
	N += abs(f[nx + offset*ny + offset2 * 0] - f[nx + offset*(ny - 1) + offset2 * 0]) / hy;
	N += abs(f[nx + offset*ny + offset2 * nz] - f[nx + offset*(ny - 1) + offset2 * nz]) / hy;

	for (unsigned int i = 1; i <= nx - 1; i++)
		N += 2 * abs(f[i + offset*ny + offset2 * 0] - f[i + offset*(ny - 1) + offset2 * 0]) / hy;

	for (unsigned int i = 1; i <= nx - 1; i++)
		N += 2 * abs(f[i + offset*ny + offset2 * nz] - f[i + offset*(ny - 1) + offset2 * nz]) / hy;

	for (unsigned int k = 1; k <= nz - 1; k++)
		N += 2 * abs(f[0 + offset*ny + offset2 * k] - f[0 + offset*(ny - 1) + offset2 * k]) / hy;
	for (unsigned int k = 1; k <= nz - 1; k++)
		N += 2 * abs(f[nx + offset*ny + offset2 * k] - f[nx + offset*(ny - 1) + offset2 * k]) / hy;


	for (unsigned int i = 1; i <= nx - 1; i++) {
		for (unsigned int k = 1; k <= nz - 1; k++) {
			N += 4 * abs(f[i + offset*ny + offset2 * k] - f[i + offset*(ny - 1) + offset2 * k]) / hy;
		}
	}

	N = N  * (hx)* (hz) / 4 / S;
	return N;
}

double Nu_x_left(double *f) {
	double N = 0;
	unsigned int offset = nx + 1;
	unsigned int offset2 = (ny + 1)*(nx + 1);
	double S = ny * hy * nz * hz;


	N += abs(f[1 + offset * 0 + offset2 * 0] - f[0 + offset * 0 + offset2 * 0]) / hx;
	N += abs(f[1 + offset*ny + offset2 * 0] - f[0 + offset*(ny)+offset2 * 0]) / hx;
	N += abs(f[1 + offset * 0 + offset2 * nz] - f[0 + offset*(0) + offset2 * nz]) / hx;
	N += abs(f[1 + offset*ny + offset2 * nz] - f[0 + offset*(ny)+offset2 * nz]) / hx;

	for (unsigned int j = 1; j <= ny - 1; j++)
		N += 2 * abs(f[1 + offset*j + offset2 * 0] - f[0 + offset*j + offset2 * 0]) / hx;

	for (unsigned int j = 1; j <= ny - 1; j++)
		N += 2 * abs(f[1 + offset*j + offset2 * nz] - f[0 + offset*j + offset2 * nz]) / hx;


	for (unsigned int k = 1; k <= nz - 1; k++)
		N += 2 * abs(f[1 + offset * 0 + offset2 * k] - f[0 + offset * 0 + offset2 * k]) / hx;
	for (unsigned int k = 1; k <= nz - 1; k++)
		N += 2 * abs(f[1 + offset*ny + offset2 * k] - f[0 + offset*ny + offset2 * k]) / hx;


	for (unsigned int j = 1; j <= ny - 1; j++) {
		for (unsigned int k = 1; k <= nz - 1; k++) {
			N += 4 * abs(f[1 + offset*j + offset2 * k] - f[0 + offset*j + offset2 * k]) / hx;
		}
	}

	N = N  * (hz)* (hy) / 4 / S;
	return N;
}
double Nu_x_right(double *f) {
	double N = 0;
	unsigned int offset = nx + 1;
	unsigned int offset2 = (ny + 1)*(nx + 1);
	double S = ny * hy * nz * hz;


	N += abs(f[nx + offset * 0 + offset2 * 0] - f[nx - 1 + offset * 0 + offset2 * 0]) / hx;
	N += abs(f[nx + offset*ny + offset2 * 0] - f[nx - 1 + offset*(ny)+offset2 * 0]) / hx;
	N += abs(f[nx + offset * 0 + offset2 * nz] - f[nx - 1 + offset*(0) + offset2 * nz]) / hx;
	N += abs(f[nx + offset*ny + offset2 * nz] - f[nx - 1 + offset*(ny)+offset2 * nz]) / hx;

	for (unsigned int j = 1; j <= ny - 1; j++)
		N += 2 * abs(f[nx + offset*j + offset2 * 0] - f[nx - 1 + offset*j + offset2 * 0]) / hx;
	for (unsigned int j = 1; j <= ny - 1; j++)
		N += 2 * abs(f[nx + offset*j + offset2 * nz] - f[nx - 1 + offset*j + offset2 * nz]) / hx;

	for (unsigned int k = 1; k <= nz - 1; k++)
		N += 2 * abs(f[nx + offset * 0 + offset2 * k] - f[nx - 1 + offset * 0 + offset2 * k]) / hx;
	for (unsigned int k = 1; k <= nz - 1; k++)
		N += 2 * abs(f[nx + offset*ny + offset2 * k] - f[nx - 1 + offset*ny + offset2 * k]) / hx;


	for (unsigned int j = 1; j <= ny - 1; j++) {
		for (unsigned int k = 1; k <= nz - 1; k++) {
			N += 4 * abs(f[nx + offset*j + offset2 * k] - f[nx - 1 + offset*j + offset2 * k]) / hy;
		}
	}

	N = N  * (hx)* (hz) / 4 / S;
	return N;
}

double Nu_z_front(double *f) {
	double N = 0;
	unsigned int offset = nx + 1;
	unsigned int offset2 = (ny + 1)*(nx + 1);
	double S = ny * hy * nx * hx;


	N += abs(f[0 + offset * 0 + offset2 * 1] - f[0 + offset * 0 + offset2 * 0]) / hz;
	N += abs(f[0 + offset*ny + offset2 * 1] - f[0 + offset*(ny)+offset2 * 0]) / hz;
	N += abs(f[nx + offset * 0 + offset2 * 1] - f[nx + offset*(0) + offset2 * 0]) / hz;
	N += abs(f[nx + offset*ny + offset2 * 1] - f[nx + offset*(ny)+offset2 * 0]) / hz;

	for (unsigned int i = 1; i <= nx - 1; i++)
		N += 2 * abs(f[i + offset*0 + offset2 * 1] - f[i + offset*0 + offset2 * 0]) / hz;
	for (unsigned int i = 1; i <= nx - 1; i++)
		N += 2 * abs(f[i + offset*ny + offset2 * 1] - f[i + offset*ny + offset2 * 0]) / hz;


	for (unsigned int j = 1; j <= ny - 1; j++)
		N += 2 * abs(f[0 + offset * j + offset2 * 1] - f[0 + offset * j + offset2 * 0]) / hz;
	for (unsigned int j = 1; j <= ny - 1; j++)
		N += 2 * abs(f[nx + offset*j + offset2 * 1] - f[nx + offset*j + offset2 * 0]) / hz;


	for (unsigned int i = 1; i <= nx - 1; i++) {
		for (unsigned int j = 1; j <= ny - 1; j++) {
			N += 4 * abs(f[i + offset*j + offset2 * 1] - f[i + offset*j + offset2 * 0]) / hz;
		}
	}

	N = N  * (hx)* (hy) / 4 / S;
	return N;
}
double Nu_z_back(double *f) {
	double N = 0;
	unsigned int offset = nx + 1;
	unsigned int offset2 = (ny + 1)*(nx + 1);
	double S = ny * hy * nx * hx;


	N += abs(f[0 + offset * 0 + offset2 * nz] - f[0 + offset * 0 + offset2 * (nz-1)]) / hz;
	N += abs(f[0 + offset*ny + offset2 * nz] - f[0 + offset*(ny)+offset2 * (nz - 1)]) / hz;
	N += abs(f[nx + offset * 0 + offset2 * nz] - f[nx + offset*(0) + offset2 * (nz - 1)]) / hz;
	N += abs(f[nx + offset*ny + offset2 * nz] - f[nx + offset*(ny)+offset2 * (nz - 1)]) / hz;

	for (unsigned int i = 1; i <= nx - 1; i++)
		N += 2 * abs(f[i + offset * 0 + offset2 * nz] - f[i + offset * 0 + offset2 * (nz - 1)]) / hz;
	for (unsigned int i = 1; i <= nx - 1; i++)
		N += 2 * abs(f[i + offset*ny + offset2 * nz] - f[i + offset*ny + offset2 * (nz - 1)]) / hz;


	for (unsigned int j = 1; j <= ny - 1; j++)
		N += 2 * abs(f[0 + offset * j + offset2 * nz] - f[0 + offset * j + offset2 * (nz - 1)]) / hz;
	for (unsigned int j = 1; j <= ny - 1; j++)
		N += 2 * abs(f[nx + offset*j + offset2 * nz] - f[nx + offset*j + offset2 * (nz - 1)]) / hz;


	for (unsigned int i = 1; i <= nx - 1; i++) {
		for (unsigned int j = 1; j <= ny - 1; j++) {
			N += 4 * abs(f[i + offset*j + offset2 * nz] - f[i + offset*j + offset2 * (nz - 1)]) / hz;
		}
	}

	N = N  * (hx)* (hy) / 4 / S;
	return N;
}



void angular_momentum(double *vx, double *vy, double *vz, unsigned int nx, unsigned int ny, unsigned int nz, double hx, double hy, double hz,
	double &AMx, double &AMy, double &AMz, double &AMabs)
{
	AMx = 0; AMy = 0; AMz = 0; AMabs = 0;
	double x0 = nx / 2 * hx;
	double y0 = ny / 2 * hy;
	double z0 = nz / 2 * hz;
	double x, y, z;
	unsigned int l;
	unsigned int offset = nx + 1;
	unsigned int offset2 = (ny + 1)*(nx + 1);

	for (int i = 1; i <= nx - 1; i++) {
		for (int j = 1; j <= ny - 1; j++) {
			for (int k = 1; k <= nz - 1; k++) {
				l = i + offset*j + offset2*k;
				x = i*hx; y = j*hy; z = k*hz;
				AMx += (z - z0) * vy[l] - (y - y0) * vz[l];
				AMy += (x - x0) * vz[l] - (z - z0) * vx[l];
				AMz += (y - y0) * vx[l] - (x - x0) * vy[l];
			}
		}
	}

	AMx *= hx*hy*hz;
	AMy *= hx*hy*hz;
	AMz *= hx*hy*hz;

	AMabs = sqrt(AMx*AMx + AMy*AMy + AMz*AMz);
}

void transform(double* f_target, double *f_source, double coef, double shift = 0) {
	int l;
	for (int i = 0; i <= nx; i++) {
		for (int j = 0; j <= ny; j++) {
			for (int k = 0; k <= nz; k++) {
				l = i + off*j + off2*k;
				f_target[l] = f_source[l] * coef + shift;
			}
		}
	}
}


#ifdef  BorderMesh
#include "deriv2.h"
#else
#include "deriv.h"
#endif //  BorderMesh


//#include "derivT.h"
__global__ void hello() {

	printf("\n thread x:%i y:%i, information copied from device:\n", threadIdx.x, threadIdx.y);

	printf("hx= %f hy=%f hz=%f \n", hx_d, hy_d, hz_d);
	printf("tau= %20.16f  \n", tau_d);
	printf("nx= %i ny=%i nz=%i N=%i \n", nx_d, ny_d, nz_d, n_d);
	printf("Lx= %f Ly=%f Lz=%f \n", Lx_d, Ly_d, Lz_d);
	printf("offset= %i offset2=%i \n", offset, offset2);

	printf("border= %i \n", border_type);

	//K_d, bettaT_d, bettaC_d, rho_d, chi_d, eta_d, grav_d, por_d, Dif_d, DifHeat_d;
	printf("K = %20.16f \n", K_d);
	printf("bettaT = %f \n", bettaT_d);
	printf("bettaC = %f \n", bettaC_d);
	printf("rho = %f \n", rho_d);
	printf("chi = %10.8f \n", chi_d);
	printf("nu = %10.8f \n", nu_d);
	printf("por = %f \n", por_d);
	printf("porB = %f \n", porB_d);
	printf("grav = %f \n", grav_d);
	printf("D = %20.16f \n", D_d);
	printf("Dt = %20.16f \n", Dt_d);
	printf("St = %20.16f \n", St_d);
	printf("Con0 = %f \n", Con0_d);
	printf("T_hot = %f \n", T_hot_d);
	printf("T_cold = %f \n", T_cold_d);
	printf("Temp0 = %f \n", Temp0_d);

	printf("rho = %f \n", rho_d);
	printf("rho_por = %f \n", rho_por_d);
	printf("Cp= %f \n", Cp_d);
	printf("Cp_por = %f \n", Cp_por_d);
	printf("rhoCpP = %f \n", rhoCpP_d);
	printf("rhoCpF = %f \n", rhoCpF_d);
	printf("Ra = %f \n", Ra_d);
	printf("Le = %f \n", Le_d);
	printf("psi = %f \n", psi_d);


	printf("\n");
}


void printParameters() {
	FILE * pFile = fopen("parameters.dat", "w");


	fprintf(pFile,"hx= %f hy=%f hz=%f \n", hx , hy , hz );
	fprintf(pFile,"tau= %20.16f  \n", tau );
	fprintf(pFile,"nx= %i ny=%i nz=%i N=%i \n", nx , ny , nz , size_l );
	fprintf(pFile,"Lx= %f Ly=%f Lz=%f \n", Lx , Ly , Lz );
	fprintf(pFile,"offset= %i offset2=%i \n", off, off2);

	fprintf(pFile,"border= %i \n", border);

	//K , bettaT , bettaC , rho , chi , eta , grav , por , Dif , DifHeat ;
	fprintf(pFile,"K = %20.16f \n", K );
	fprintf(pFile,"bettaT = %f \n", bettaT );
	fprintf(pFile,"bettaC = %f \n", bettaC );
	fprintf(pFile,"rho = %f \n", rho );
	fprintf(pFile,"chi = %10.8f \n", chi );
	fprintf(pFile,"nu = %10.8f \n", nu );
	fprintf(pFile,"por = %f \n", por );
	
	fprintf(pFile,"grav = %f \n", grav );
	fprintf(pFile,"D = %20.16f \n", D );
	fprintf(pFile,"Dt = %20.16f \n", Dt );
	fprintf(pFile,"St = %20.16f \n", St );
	fprintf(pFile,"Con0 = %f \n", Con0 );
	fprintf(pFile,"T_hot = %f \n", T_hot );
	fprintf(pFile,"T_cold = %f \n", T_cold );
	fprintf(pFile,"Temp0 = %f \n", Temp0 );

	fprintf(pFile,"rho = %f \n", rho );
	fprintf(pFile,"rho_por = %f \n", rho_por );
	fprintf(pFile,"Cp= %f \n", Cp );
	fprintf(pFile,"Cp_por = %f \n", Cp_por );
	fprintf(pFile,"rhoCpP = %f \n", rhoCpP );
	fprintf(pFile,"rhoCpF = %f \n", rhoCpF );
	fprintf(pFile,"Ra = %f \n", Ra );
	fprintf(pFile,"Le = %f \n", Le );
	fprintf(pFile,"psi = %f \n", psi );
	fprintf(pFile, "porB = %f \n", porB);

	fprintf(pFile,"\n");

	fclose(pFile);
}

/*projection method*/
//__global__ void quasi_velocity(double *ux_d, double *uy_d, double *uz_d, double *vx_d, double *vy_d, double *vz_d, double *T_d, double *T0_d, double *C_d, double *C0_d) {
//
//	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
//	unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
//	unsigned int k = threadIdx.z + blockIdx.z*blockDim.z;
//	unsigned int l = i + offset*j + offset2*k;
//	if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d)
//	{
//
//		/*
//		INNER
//		*/
//		if (i > 0 && i < nx_d && j > 0 && j < ny_d && k > 0 && k < nz_d)
//		{
//
//			//ux_d
//			ux_d[l] = vx_d[l]
//				+ tau_d * (
//#ifdef NV_YES
//				((-vx_d[l] * dx1(vx_d, l, i) - vy_d[l] * dy1(vx_d, l, j) - vz_d[l] * dz1(vx_d, l, k))/por_d + nu_d * laplace(vx_d, l, i, j, k))
//#endif
//
//
//					- nu_d / K_d * vx_d[l]  * por_d
//					);
//
//			//uy_d
//			uy_d[l] = vy_d[l]
//				+ tau_d  * (
//#ifdef NV_YES
//				((-vx_d[l] * dx1(vy_d, l, i) - vy_d[l] * dy1(vy_d, l, j) - vz_d[l] * dz1(vy_d, l, k)) / por_d + nu_d * laplace(vy_d, l, i, j, k))
//#endif
//
//					- nu_d / K_d * vy_d[l] * por_d
//
//					+ grav_d*(bettaT_d*(T0_d[l] - Temp0_d) + bettaC_d*(C0_d[l] - Con0_d)) * por_d
//					);
//
//			//uz_d
//			uz_d[l] = vz_d[l]
//				+ tau_d * (
//#ifdef NV_YES
//				((-vx_d[l] * dx1(vz_d, l, i) - vy_d[l] * dy1(vz_d, l, j) - vz_d[l] * dz1(vz_d, l, k)) / por_d + nu_d * laplace(vz_d, l, i, j, k))
//#endif
//
//
//					- nu_d / K_d * vz_d[l]  * por_d
//
//					);
//
//		}
//
//		/*
//		UP-DOWN
//		*/
//
//		// y
//		else if (j == 0 && (i > 0 && i < nx_d && k > 0 && k < nz_d))
//		{
//			uy_d[l] =  tau_d * grav_d*(bettaT_d*(T0_d[l] - Temp0_d) + bettaC_d*(C0_d[l] - Con0_d)) * por_d;
//#ifdef NV_YES
//			uy_d[l] += nu_d * tau_d * dy2_up(l, vy_d);
//#endif
//			//uy_d[l] = tau_d / hy_d / hy_d*(2.0 * vy_d[l] - 5.0 * vy_d[l + offset] + 4.0 * vy_d[l + offset * 2] - vy_d[l + offset * 3]) + tau_d* Ra_d / Pr_d*(T0_d[l] + C0_d[l]);
//			//ux_d[l] = tau_d / hy_d / hy_d*(2 * vx_d[l] - 5 * vx_d[l + offset] + 4 * vx_d[l + offset * 2] - vx_d[l + offset * 3]);
//			//uz_d[l] = tau_d / hz_d / hz_d*(2 * vz_d[l] - 5 * vz_d[l + offset] + 4 * vz_d[l + offset * 2] - vz_d[l + offset * 3]);
//		}
//		else if (j == ny_d && (i > 0 && i < nx_d && k > 0 && k < nz_d))
//		{
//			uy_d[l] =  tau_d * grav_d*(bettaT_d*(T0_d[l] - Temp0_d) + bettaC_d*(C0_d[l] - Con0_d)) * por_d;
//#ifdef NV_YES
//			uy_d[l] += nu_d * tau_d * dy2_down(l, vy_d);
//#endif 
//
//			//uy_d[l] = tau_d / hy_d / hy_d*(2.0 * vy_d[l] - 5.0 * vy_d[l - offset] + 4.0 * vy_d[l - offset * 2] - vy_d[l - offset * 3]) + tau_d*Ra_d / Pr_d*(T0_d[l] + C0_d[l]);
//			//ux_d[l] = tau_d / hy_d / hy_d*(2 * vx_d[l] - 5 * vx_d[l - offset] + 4 * vx_d[l - offset * 2] - vx_d[l - offset * 3]);
//			//uz_d[l] = tau_d / hy_d / hy_d*(2 * vz_d[l] - 5 * vz_d[l - offset] + 4 * vz_d[l - offset * 2] - vz_d[l - offset * 3]);
//		}
//
//		/*
//		CLOSED
//		*/
//
//		// x
//		else if (border_type == 0 && i == 0 && (j > 0 && j < ny_d && k > 0 && k < nz_d))
//		{
//			ux_d[l] = 0;  //-tau_d * grav_d*(bettaT_d*(T0_d[l] - Temp0_d) + bettaC_d*(C0_d[l] - Con0_d)) * por_d;
//#ifdef NV_YES
//			ux_d[l] += nu_d * tau_d * dx2_forward(l, vx_d);
//#endif 
//			//ux_d[l] = tau_d / hx_d / hx_d * (2.0 * vx_d[l] - 5.0 * vx_d[l + 1] + 4.0 * vx_d[l + 2] - vx_d[l + 3]) + tau_d*Ra_d / Pr_d*(T0_d[l] + C10_d[l] + C20_d[l]) * sinA;
//			//uy_d[l] = tau_d / hx_d / hx_d * (2 * vy_d[l] - 5 * vy_d[l + 1] + 4 * vy_d[l + 2] - vy_d[l + 3]);
//			//uz_d[l] = tau_d / hx_d / hx_d * (2 * vz_d[l] - 5 * vz_d[l + 1] + 4 * vz_d[l + 2] - vz_d[l + 3]);
//		}
//		else if (border_type == 0 && i == nx_d && (j > 0 && j < ny_d && k > 0 && k < nz_d))
//		{
//			ux_d[l] = 0; // -tau_d * grav_d*(bettaT_d*(T0_d[l] - Temp0_d) + bettaC_d*(C0_d[l] - Con0_d)) * por_d;
//#ifdef NV_YES
//			ux_d[l] += nu_d * tau_d * dx2_back(l, vx_d);
//#endif 
//			//ux_d[l] = tau_d / hx_d / hx_d * (2.0 * vx_d[l] - 5.0 * vx_d[l - 1] + 4.0 * vx_d[l - 2] - vx_d[l - 3]) + tau_d*Ra_d / Pr_d*(T0_d[l] + C10_d[l] + C20_d[l]) * sinA;
//			//uy_d[l] = tau_d / hx_d / hx_d * (2 * vy_d[l] - 5 * vy_d[l - 1] + 4 * vy_d[l - 2] - vy_d[l - 3]);
//			//uz_d[l] = tau_d / hx_d / hx_d * (2 * vz_d[l] - 5 * vz_d[l - 1] + 4 * vz_d[l - 2] - vz_d[l - 3]);
//		}
//
//		// z
//		else if (border_type == 0 && k == 0 && (i > 0 && i < nx_d && j > 0 && j < ny_d))
//		{
//			uz_d[l] = 0; // -tau_d * grav_d*(bettaT_d*(T0_d[l] - Temp0_d) + bettaC_d*(C0_d[l] - Con0_d)) * por_d;
//#ifdef NV_YES
//			uz_d[l] += nu_d * tau_d * dz2_toDeep(l, vz_d);
//#endif 
//			//ux_d[l] = tau_d / hz_d / hz_d * (2 * vx_d[l] - 5 * vx_d[l + offset2] + 4 * vx_d[l + offset2 * 2] - vx_d[l + offset2 * 3]);
//			//uy_d[l] = tau_d / hz_d / hz_d * (2 * vy_d[l] - 5 * vy_d[l + offset2] + 4 * vy_d[l + offset2 * 2] - vy_d[l + offset2 * 3]);
//			//uz_d[l] = tau_d / hz_d / hz_d * (2.0 * vz_d[l] - 5.0 * vz_d[l + offset2] + 4.0 * vz_d[l + offset2 * 2] - vz_d[l + offset2 * 3]);
//		}
//		else if (border_type == 0 && k == nz_d && (i > 0 && i < nx_d && j > 0 && j < ny_d))
//		{
//			uz_d[l] = 0; // -tau_d * grav_d*(bettaT_d*(T0_d[l] - Temp0_d) + bettaC_d*(C0_d[l] - Con0_d)) * por_d;
//#ifdef NV_YES
//			uz_d[l] += nu_d * tau_d * dz2_toUs(l, vz_d);
//#endif 
//			//ux_d[l] = tau_d / hz_d / hz_d * (2 * vx_d[l] - 5 * vx_d[l - offset2] + 4 * vx_d[l - offset2 * 2] - vx_d[l - offset2 * 3]);
//			//uy_d[l] = tau_d / hz_d / hz_d * (2 * vy_d[l] - 5 * vy_d[l - offset2] + 4 * vy_d[l - offset2 * 2] - vy_d[l - offset2 * 3]);
//			//uz_d[l] = tau_d / hz_d / hz_d * (2.0 * vz_d[l] - 5.0 * vz_d[l - offset2] + 4.0 * vz_d[l - offset2 * 2] - vz_d[l - offset2 * 3]);
//		}
//
//		// corner points
//
//		else if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d) {
//			int ii = i + 1 - i / nx_d - ceil(i / (nx_d + 1.0));
//			int jj = j + 1 - j / ny_d - ceil(j / (ny_d + 1.0));
//			int kk = k + 1 - k / nz_d - ceil(k / (nz_d + 1.0));
//
//			//ux_d[l] = 0;
//			//uy_d[l] = 0;
//			//uz_d[l] = 0;
//
//			//ux_d[l] = ux_d[ii + offset*jj + offset2*kk];
//			//uy_d[l] = uy_d[ii + offset*jj + offset2*kk];
//			//uz_d[l] = uz_d[ii + offset*jj + offset2*kk];
//		}
//
//
//	}
//}
//__global__ void velocity_correction(double *ux_d, double *uy_d, double *uz_d, double *vx_d, double *vy_d, double *vz_d, double *p_d) {
//
//
//	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
//	unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
//	unsigned int k = threadIdx.z + blockIdx.z*blockDim.z;
//	unsigned int l = i + offset*j + offset2*k;
//
//	if (border_type == 0 && i <= nx_d && j <= ny_d && k <= nz_d && l < n_d)
//	{
//		if (i > 0 && i < nx_d && j > 0 && j < ny_d && k > 0 && k < nz_d)
//		{
//			vx_d[l] = ux_d[l] - tau_d / (2.0 * hx_d)*(p_d[l + 1] - p_d[l - 1]) * por_d / rho_d;
//			vy_d[l] = uy_d[l] - tau_d / (2.0 * hy_d)*(p_d[l + offset] - p_d[l - offset]) * por_d / rho_d;
//			vz_d[l] = uz_d[l] - tau_d / (2.0 * hz_d)*(p_d[l + offset2] - p_d[l - offset2]) * por_d / rho_d;
//		}
//		else if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d) {
//			vx_d[l] = 0; vy_d[l] = 0; 	vz_d[l] = 0;
//		}
//
//	}
//
//
//	else if (border_type == 1 && i <= nx_d && j <= ny_d && k <= nz_d && l < n_d)
//	{
//		if (i > 0 && i < nx_d && j > 0 && j < ny_d && k > 0 && k < nz_d)
//		{
//			vx_d[l] = ux_d[l] - tau_d / 2.0 / hx_d*(p_d[l + 1] - p_d[l - 1]);
//			vy_d[l] = uy_d[l] - tau_d / 2.0 / hy_d*(p_d[l + offset] - p_d[l - offset]);
//			vz_d[l] = uz_d[l] - tau_d / 2.0 / hz_d*(p_d[l + offset2] - p_d[l - offset2]);
//		}
//		else if (j == 0 && (i > 0 && i < nx_d && k > 0 && k < nz_d))
//		{
//			vx_d[l] = 0.0; vy_d[l] = 0.0; vz_d[l] = 0.0;
//		}
//		else if (j == ny_d && (i > 0 && i < nx_d && k > 0 && k < nz_d))
//		{
//			vx_d[l] = 0.0; vy_d[l] = 0.0; vz_d[l] = 0.0;
//		}
//		else if (i == 0 && (j > 0 && j < ny_d && k > 0 && k < nz_d))
//		{
//			int ll = nx_d - 1 + offset*j + offset2*k;
//			vx_d[l] = ux_d[ll] - tau_d / 2.0 / hx_d*(p_d[ll + 1] - p_d[ll - 1]);
//			vy_d[l] = uy_d[ll] - tau_d / 2.0 / hy_d*(p_d[ll + offset] - p_d[ll - offset]);
//			vz_d[l] = uz_d[ll] - tau_d / 2.0 / hz_d*(p_d[ll + offset2] - p_d[ll - offset2]);
//		}
//		else if (i == nx_d && (j > 0 && j < ny_d && k > 0 && k < nz_d))
//		{
//			int ll = 1 + offset*j + offset2*k;
//			vx_d[l] = ux_d[ll] - tau_d / 2.0 / hx_d*(p_d[ll + 1] - p_d[ll - 1]);
//			vy_d[l] = uy_d[ll] - tau_d / 2.0 / hy_d*(p_d[ll + offset] - p_d[ll - offset]);
//			vz_d[l] = uz_d[ll] - tau_d / 2.0 / hz_d*(p_d[ll + offset2] - p_d[ll - offset2]);
//		}
//		else if (k == 0 && (i > 0 && i < nx_d && j > 0 && j < ny_d))
//		{
//			int ll = i + offset*j + offset2*nz_d - offset2;
//			vx_d[l] = ux_d[ll] - tau_d / 2.0 / hx_d*(p_d[ll + 1] - p_d[ll - 1]);
//			vy_d[l] = uy_d[ll] - tau_d / 2.0 / hy_d*(p_d[ll + offset] - p_d[ll - offset]);
//			vz_d[l] = uz_d[ll] - tau_d / 2.0 / hz_d*(p_d[ll + offset2] - p_d[ll - offset2]);
//		}
//		else if (k == nz_d && (i > 0 && i < nx_d && j > 0 && j < ny_d))
//		{
//			int ll = i + offset*j + offset2;
//			vx_d[l] = ux_d[ll] - tau_d / 2.0 / hx_d*(p_d[ll + 1] - p_d[ll - 1]);
//			vy_d[l] = uy_d[ll] - tau_d / 2.0 / hy_d*(p_d[ll + offset] - p_d[ll - offset]);
//			vz_d[l] = uz_d[ll] - tau_d / 2.0 / hz_d*(p_d[ll + offset2] - p_d[ll - offset2]);
//		}
//	}
//
//	else if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d) {
//		int ii = i + 1 - i / nx_d - ceil(i / (nx_d + 1.0));
//		int jj = j + 1 - j / ny_d - ceil(j / (ny_d + 1.0));
//		int kk = k + 1 - k / nz_d - ceil(k / (nz_d + 1.0));
//
//		vx_d[l] = 0;
//		vy_d[l] = 0;
//		vz_d[l] = 0;
//
//	}
//
//}
//__global__ void Poisson(double *ux_d, double *uy_d, double *uz_d, double *p_d, double *p0_d)
//{
//	//период условие отключено
//	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
//	unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
//	unsigned int k = threadIdx.z + blockIdx.z*blockDim.z;
//	unsigned int l = i + offset*j + offset2*k;
//	//double psiav0 = 0.0; double psiav = 0.0; double eps = 1.0; int k = 0;
//	
//	/*
//
//	closed
//
//	*/
//
//	if (border_type == 0 && i <= nx_d && j <= ny_d && k <= nz_d && l < n_d)
//	{
//		if (i > 0 && i < nx_d && j > 0 && j < ny_d && k > 0 && k < nz_d)
//		{
//			p_d[l] =
//				-(dx1(ux_d, l, i) + dy1(uy_d, l, j) + dz1(uz_d, l, k)) / tau_d * rho_d / por_d;
//
//
//			if (i == 1) 				p_d[l] += 2.0 / 3.0 / hx_d / hx_d*(p0_d[l + 1] - p0_d[l] - hx_d / tau_d*ux_d[l - 1] * rho_d / por_d);
//			else if (i == nx_d - 1) 	p_d[l] += 2.0 / 3.0 / hx_d / hx_d*(p0_d[l - 1] - p0_d[l] + hx_d / tau_d*ux_d[l + 1] * rho_d / por_d);
//			else						
//				p_d[l] += 1.0 / hx_d / hx_d*(p0_d[l + 1] + p0_d[l - 1] - 2.0*p0_d[l]);
//
//			if (j == 1)					p_d[l] += 2.0 / 3.0 / hy_d / hy_d*(p0_d[l + offset] - p0_d[l] - hy_d / tau_d*uy_d[l - offset] * rho_d / por_d);
//			else if (j == ny_d - 1) 	p_d[l] += 2.0 / 3.0 / hy_d / hy_d*(p0_d[l - offset] - p0_d[l] + hy_d / tau_d*uy_d[l + offset] * rho_d / por_d);
//			else						
//				p_d[l] += 1.0 / hy_d / hy_d*(p0_d[l + offset] + p0_d[l - offset] - 2.0*p0_d[l]);
//
//			if (k == 1)					p_d[l] += 2.0 / 3.0 / hz_d / hz_d*(p0_d[l + offset2] - p0_d[l] - hz_d / tau_d*uz_d[l - offset2] * rho_d / por_d);
//			else if (k == nz_d - 1)		p_d[l] += 2.0 / 3.0 / hz_d / hz_d*(p0_d[l - offset2] - p0_d[l] + hz_d / tau_d*uz_d[l + offset2] * rho_d / por_d);
//			else						
//				p_d[l] += 1.0 / hz_d / hz_d*(p0_d[l + offset2] + p0_d[l - offset2] - 2.0*p0_d[l]);
//
//			p_d[l] *= tau_p_d;
//			p_d[l] += p0_d[l];
//		}
//
//
//		else if (j == 0 && (i > 0 && i < nx_d && k > 0 && k < nz_d))		p_d[l] = (4.0*p0_d[l + offset] - p0_d[l + offset * 2]) / 3.0 - uy_d[l] * rho_d / por_d * 2.0 * hy_d / tau_d / 3.0;
//		else if (j == ny_d && (i > 0 && i < nx_d && k > 0 && k < nz_d))		p_d[l] = (4.0*p0_d[l - offset] - p0_d[l - offset * 2]) / 3.0 + uy_d[l] * rho_d / por_d * 2.0 * hy_d / tau_d / 3.0;
//		else if (i == 0 && (j > 0 && j < ny_d && k > 0 && k < nz_d))		p_d[l] = (4.0*p0_d[l + 1] - p0_d[l + 2]) / 3.0 - ux_d[l] * rho_d / por_d * 2.0 * hx_d / tau_d / 3.0;
//		else if (i == nx_d && (j > 0 && j < ny_d && k > 0 && k < nz_d))		p_d[l] = (4.0*p0_d[l - 1] - p0_d[l - 2]) / 3.0 + ux_d[l] * rho_d / por_d * 2.0 * hx_d / tau_d / 3.0;
//		else if (k == 0 && (i > 0 && i < nx_d && j > 0 && j < ny_d))		p_d[l] = (4.0*p0_d[l + offset2] - p0_d[l + offset2 * 2]) / 3.0 - uz_d[l] * rho_d / por_d * 2.0 * hz_d / tau_d / 3.0;
//		else if (k == nz_d && (i > 0 && i < nx_d && j > 0 && j < ny_d))		p_d[l] = (4.0*p0_d[l - offset2] - p0_d[l - offset2 * 2]) / 3.0 + uz_d[l] * rho_d / por_d * 2.0 * hz_d / tau_d / 3.0;
//
//		else if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d) {
//			int ii = i + 1 - i / nx_d - ceil(i / (nx_d + 1.0));
//			int jj = j + 1 - j / ny_d - ceil(j / (ny_d + 1.0));
//			int kk = k + 1 - k / nz_d - ceil(k / (nz_d + 1.0));
//			p_d[l] = p0_d[ii + offset*jj + offset2*kk];
//		}
//	}
//}


#ifdef level3
__global__ void HeatEquation(double *vx_d, double *vy_d, double *vz_d, double *T_d, double *T0_d) {


	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z*blockDim.z;
	unsigned int l = i + offset*j + offset2*k;
	if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d)
	{

		/*
		INNER
		*/
		if (i > 0 && i < nx_d && j > 0 && j < ny_d && k > 0 && k < nz_d)
		{

			//phi
			T_d[l] = T0_d[l]
				+ tau_d * (
					+(-vx_d[l] * dx1(T0_d, l, i)
						- vy_d[l] * dy1(T0_d, l, j)
						- vz_d[l] * dz1(T0_d, l, k))*(rhoCpF_d) / (rhoCpP_d)

					+chi_d*laplace(T0_d, l, i, j, k)

					);

		}



		/*
		UP-DOWN
		*/

		// y
		else if (j == 0 && (i > 0 && i < nx_d && k > 0 && k < nz_d)) 		T_d[l] = dy1_eq_0_up(l, T0_d); // T_hot_d;  //  
		else if (j == ny_d && (i > 0 && i < nx_d && k > 0 && k < nz_d))		T_d[l] = dy1_eq_0_down(l, T0_d); //T_cold_d; //   

																											 /*
																											 CLOSED
																											 */

																											 // x
		else if (border_type == 0 && i == 0 && (j > 0 && j < ny_d && k > 0 && k < nz_d)) 		T_d[l] = dx1_eq_0_forward(l, T0_d);
		else if (border_type == 0 && i == nx_d && (j > 0 && j < ny_d && k > 0 && k < nz_d))		T_d[l] = dx1_eq_0_back(l, T0_d);

		// z
		else if (border_type == 0 && k == 0 && (i > 0 && i < nx_d && j > 0 && j < ny_d))		T_d[l] = T_hot_d;  //   dz1_eq_0_toDeep(l, T0_d); //
		else if (border_type == 0 && k == nz_d && (i > 0 && i < nx_d && j > 0 && j < ny_d))		T_d[l] = T_cold_d; // dz1_eq_0_toUs(l, T0_d); //

																												   // corner points

		else if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d) {

			int ii = i + 1 - i / nx_d - ceil(i / (nx_d + 1.0));
			int jj = j + 1 - j / ny_d - ceil(j / (ny_d + 1.0));
			int kk = k + 1 - k / nz_d - ceil(k / (nz_d + 1.0));

			T_d[l] = T0_d[ii + offset*jj + offset2*kk];
		}


	}
}
__global__ void Concentration(double *vx_d, double *vy_d, double *vz_d, double *T_d, double *T0_d, double *C_d, double *C0_d) {


	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z*blockDim.z;
	unsigned int l = i + offset*j + offset2*k;
	if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d)
	{

		/*
		INNER
		*/
		if (i > 0 && i < nx_d && j > 0 && j < ny_d && k > 0 && k < nz_d)
		{

			//phi
			C_d[l] = C0_d[l]
				+ tau_d / porB_d * (
					-vx_d[l] * dx1(C0_d, l, i)
					- vy_d[l] * dy1(C0_d, l, j)
					- vz_d[l] * dz1(C0_d, l, k)

					+ (D_d*laplace(C0_d, l, i, j, k) - psi_d * Dt_d*laplace(T0_d, l, i, j, k)) / Le_d

					);

		}



		/*
		UP-DOWN
		*/

		// y
		else if (j == 0 && (i > 0 && i < nx_d && k > 0 && k < nz_d)) 		C_d[l] = dy1_eq_0_up(l, C0_d);
		else if (j == ny_d && (i > 0 && i < nx_d && k > 0 && k < nz_d))		C_d[l] = dy1_eq_0_down(l, C0_d);

		/*
		CLOSED
		*/

		// x
		else if (i == 0 && (j > 0 && j < ny_d && k > 0 && k < nz_d)) 		C_d[l] = dx1_eq_0_forward(l, C0_d);
		else if (i == nx_d && (j > 0 && j < ny_d && k > 0 && k < nz_d))		C_d[l] = dx1_eq_0_back(l, C0_d);

		// z
		else if (k == 0 && (i > 0 && i < nx_d && j > 0 && j < ny_d))		C_d[l] = (4.0*C0_d[l + offset2] - C0_d[l + offset2 * 2]) / 3.0 + (psi_d * St_d  * (3.0*T0_d[l] - 4.0*T0_d[l + offset2] + T0_d[l + offset2 * 2])) / 3.0;
		else if (k == nz_d && (i > 0 && i < nx_d && j > 0 && j < ny_d))		C_d[l] = (4.0*C0_d[l - offset2] - C0_d[l - offset2 * 2]) / 3.0 + (psi_d * St_d  * (3.0*T0_d[l] - 4.0*T0_d[l - offset2] + T0_d[l - offset2 * 2])) / 3.0;
		/*psi = -1 if dimensional*/
		// corner points

		else if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d) {

			int ii = i + 1 - i / nx_d - ceil(i / (nx_d + 1.0));
			int jj = j + 1 - j / ny_d - ceil(j / (ny_d + 1.0));
			int kk = k + 1 - k / nz_d - ceil(k / (nz_d + 1.0));

			C_d[l] = C0_d[ii + offset*jj + offset2*kk];
		}


	}
}
__global__ void Poisson_Darcy(double *p_d, double *p0_d, double *T0_d, double *C0_d)
{
	//период условие отключено
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z*blockDim.z;
	unsigned int l = i + offset*j + offset2*k;
	//double psiav0 = 0.0; double psiav = 0.0; double eps = 1.0; int k = 0;


	double coef = Ra_d * grav_d * rho_d;
	if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d)
	{
		if (i > 0 && i < nx_d && j > 0 && j < ny_d && k > 0 && k < nz_d)
		{
			p_d[l] = -coef*(bettaT_d*dy1_center(l, T0_d) + bettaC_d*dy1_center(l, C0_d));

			p_d[l] += dx2_center(l, p0_d);
			p_d[l] += dy2_center(l, p0_d);
			p_d[l] += dz2_center(l, p0_d);

			p_d[l] *= tau_p_d;
			p_d[l] += p0_d[l];
		}


		else if (j == 0 && (i > 0 && i < nx_d && k > 0 && k < nz_d))		p_d[l] = (4.0*p0_d[l + offset] - p0_d[l + offset * 2]) / 3.0 - coef*(bettaT_d*(T0_d[l] - Temp0_d) + bettaC_d *(C0_d[l] - Con0_d)) * (2.0 * hy_d) / 3.0;
		else if (j == ny_d && (i > 0 && i < nx_d && k > 0 && k < nz_d))		p_d[l] = (4.0*p0_d[l - offset] - p0_d[l - offset * 2]) / 3.0 + coef*(bettaT_d*(T0_d[l] - Temp0_d) + bettaC_d *(C0_d[l] - Con0_d)) * (2.0 * hy_d) / 3.0;
		else if (i == 0 && (j > 0 && j < ny_d && k > 0 && k < nz_d))		p_d[l] = dx1_eq_0_forward(l, p0_d);
		else if (i == nx_d && (j > 0 && j < ny_d && k > 0 && k < nz_d))		p_d[l] = dx1_eq_0_back(l, p0_d);
		else if (k == 0 && (i > 0 && i < nx_d && j > 0 && j < ny_d))		p_d[l] = dz1_eq_0_toDeep(l, p0_d);
		else if (k == nz_d && (i > 0 && i < nx_d && j > 0 && j < ny_d))		p_d[l] = dz1_eq_0_toUs(l, p0_d);

		else if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d) {
			int ii = i + 1 - i / nx_d - ceil(i / (nx_d + 1.0));
			int jj = j + 1 - j / ny_d - ceil(j / (ny_d + 1.0));
			int kk = k + 1 - k / nz_d - ceil(k / (nz_d + 1.0));
			p_d[l] = p0_d[ii + offset*jj + offset2*kk];
		}
	}
}
__global__ void Velocity_Darcy(double *vx_d, double *vy_d, double *vz_d, double *p0_d, double *T0_d, double *C0_d) {
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z*blockDim.z;
	unsigned int l = i + offset*j + offset2*k;

	/*
	closed
	*/
	double coef = Ra_d*grav_d;
	double coef2 = K_d / nu_d;
	if (border_type == 0 && i <= nx_d && j <= ny_d && k <= nz_d && l < n_d)
	{
		if (i > 0 && i < nx_d && j > 0 && j < ny_d && k > 0 && k < nz_d)
		{
			vy_d[l] = (-dy1_center(l, p0_d) / rho_d + coef*(bettaT_d*(T0_d[l] - Temp0_d) + bettaC_d*(C0_d[l] - Con0_d)))* coef2;
			vx_d[l] = (-dx1_center(l, p0_d) / rho_d) * coef2;
			vz_d[l] = (-dz1_center(l, p0_d) / rho_d) * coef2;
		}


		else if (j == 0 && (i > 0 && i < nx_d && k > 0 && k < nz_d))		vy_d[l] = 0;
		else if (j == ny_d && (i > 0 && i < nx_d && k > 0 && k < nz_d))		vy_d[l] = 0;
		else if (i == 0 && (j > 0 && j < ny_d && k > 0 && k < nz_d))		vx_d[l] = 0;
		else if (i == nx_d && (j > 0 && j < ny_d && k > 0 && k < nz_d))		vx_d[l] = 0;
		else if (k == 0 && (i > 0 && i < nx_d && j > 0 && j < ny_d))		vz_d[l] = 0;
		else if (k == nz_d && (i > 0 && i < nx_d && j > 0 && j < ny_d))		vz_d[l] = 0;

		else if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d) {
			vx_d[l] = vy_d[l] = vz_d[l] = 0;
		}
	}
}

#endif // level3


#ifdef level2
__global__ void HeatEquation(double *vx_d, double *vy_d, double *vz_d, double *T_d, double *T0_d) {


	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z*blockDim.z;
	unsigned int l = i + offset*j + offset2*k;
	if (i <= nx_d && j <= ny_d && l < n_d)
	{

		/*
		INNER
		*/
		if (i > 0 && i < nx_d && j > 0 && j < ny_d)
		{

			//phi
			T_d[l] = T0_d[l]
				+ tau_d * (
					+(-vx_d[l] * dx1(T0_d, l, i)
						- vy_d[l] * dy1(T0_d, l, j)
						)*(rhoCpF_d) / (rhoCpP_d)

					+chi_d*laplace(T0_d, l, i, j)

					);

		}



		/*
		UP-DOWN
		*/

		// y
		else if (j == 0 && (i > 0 && i < nx_d)) 		T_d[l] = dy1_eq_0_up(l, T0_d, j); // T_hot_d;  //  
		else if (j == ny_d && (i > 0 && i < nx_d))		T_d[l] = dy1_eq_0_down(l, T0_d, j); //T_cold_d; //   

		else if (i == 0) 		T_d[l] = T_hot_d;
		else if (i == nx_d)		T_d[l] = T_cold_d;
		// corner points




	}
}
__global__ void Concentration(double *vx_d, double *vy_d, double *vz_d, double *T_d, double *T0_d, double *C_d, double *C0_d) {


	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z*blockDim.z;
	unsigned int l = i + offset*j + offset2*k;
	if (i <= nx_d && j <= ny_d && l < n_d)
	{

		/*
		INNER
		*/
		if (i > 0 && i < nx_d && j > 0 && j < ny_d)
		{

			//phi
			C_d[l] = C0_d[l]
				+ tau_d / porB_d * (
					-vx_d[l] * dx1(C0_d, l, i)
					- vy_d[l] * dy1(C0_d, l, j)
					//-VgradF(vx_d, vy_d, C0_d, l, i, j)

					+ (D_d*laplace(C0_d, l, i, j) - psi_d * Dt_d*laplace(T0_d, l, i, j)) / Le_d

					);

		}

		else if (j == 0 && (i > 0 && i < nx_d)) 		C_d[l] = dy1_eq_0_up(l, C0_d, j);
		else if (j == ny_d && (i > 0 && i < nx_d))		C_d[l] = dy1_eq_0_down(l, C0_d, j);
		//else if (j == 0 && (i > 0 && i < nx_d)) 		C_d[l] = (4.0*C0_d[l + offset] - C0_d[l + 2*offset]) / 3.0 + (psi_d * St_d  * (3.0*T0_d[l] - 4.0*T0_d[l + offset] + T0_d[l + 2*offset])) / 3.0;
		//else if (j == ny_d && (i > 0 && i < nx_d))		C_d[l] = (4.0*C0_d[l - offset] - C0_d[l - 2*offset]) / 3.0 + (psi_d * St_d  * (3.0*T0_d[l] - 4.0*T0_d[l - offset] + T0_d[l - 2*offset])) / 3.0;
		else if (i == 0) 		C_d[l] = (4.0*C0_d[l + 1] - C0_d[l + 2]) / 3.0 + (psi_d * St_d  * (3.0*T0_d[l] - 4.0*T0_d[l + 1] + T0_d[l + 2])) / 3.0;
		else if (i == nx_d)		C_d[l] = (4.0*C0_d[l - 1] - C0_d[l - 2]) / 3.0 + (psi_d * St_d  * (3.0*T0_d[l] - 4.0*T0_d[l - 1] + T0_d[l - 2])) / 3.0;
		/*psi = -1 if dimensional*/

// dx1_eq_0_forward(l, C0_d); // Con0_d*1.0; //
// dx1_eq_0_back(l, C0_d); // Con0_d*1.0; //
	}
}
#ifdef Stream
__global__ void Poisson_Darcy(double *ksi, double *ksi0, double *T0_d, double *C0_d)
{
	//период условие отключено
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z*blockDim.z;
	unsigned int l = i + offset*j + offset2*k;
	//double psiav0 = 0.0; double psiav = 0.0; double eps = 1.0; int k = 0;


	double coef = Ra_d * grav_d * K_d / nu_d;
	if (i <= nx_d && j <= ny_d && l < n_d)
	{
		if (i > 0 && i < nx_d && j > 0 && j < ny_d)
		{
			ksi[l] = coef*(bettaT_d*dx1(T0_d, l, i) + bettaC_d*dx1(C0_d, l, i));

			ksi[l] += dx2(ksi0, l, i);
			ksi[l] += dy2(ksi0, l, j);

			ksi[l] *= tau_p_d;
			ksi[l] += ksi0[l];
		}


		else if (j == 0 && (i > 0 && i < nx_d))		ksi[l] = 0;
		else if (j == ny_d && (i > 0 && i < nx_d))		ksi[l] = 0;
		else if (i == 0 && (j > 0 && j < ny_d))		ksi[l] = 0;
		else if (i == nx_d && (j > 0 && j < ny_d))		ksi[l] = 0;


		else if (i <= nx_d && j <= ny_d && k <= nz_d && l < n_d) {
			ksi0[l] = 0;
		}
	}
}
__global__ void Velocity_Darcy(double *vx_d, double *vy_d, double *vz_d, double *ksi, double *T0_d, double *C0_d) {
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int k = threadIdx.z + blockIdx.z*blockDim.z;
	unsigned int l = i + offset*j + offset2*k;


	if (i <= nx_d && j <= ny_d && l < n_d)
	{
		if (i > 0 && i < nx_d && j > 0 && j < ny_d)
		{
			vy_d[l] = -dx1(ksi, l, i);
			vx_d[l] = dy1(ksi, l, j);
		}


		else if (j == 0 && (i > 0 && i < nx_d))		vy_d[l] = 0;
		else if (j == ny_d && (i > 0 && i < nx_d))		vy_d[l] = 0;
		else if (i == 0 && (j > 0 && j < ny_d))		vx_d[l] = 0;
		else if (i == nx_d && (j > 0 && j < ny_d))		vx_d[l] = 0;

		else if (i <= nx_d && j <= ny_d && l < n_d) {
			vx_d[l] = vy_d[l] = 0;
		}
	}
}
#else
__global__ void Poisson_Darcy(double *p_d, double *p0_d, double *T0_d, double *C0_d)
{
	//период условие отключено
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int l = i + offset*j;
	//double psiav0 = 0.0; double psiav = 0.0; double eps = 1.0; int k = 0;


	double coef = Ra_d * grav_d * rho_d;
	if (i <= nx_d && j <= ny_d && l < n_d)
	{
		if (i > 0 && i < nx_d && j > 0 && j < ny_d)
		{
			p_d[l] = -coef*(bettaT_d*dy1_center(l, T0_d) + bettaC_d*dy1_center(l, C0_d));

			p_d[l] += dx2_center(l, p0_d);
			p_d[l] += dy2_center(l, p0_d);

			//if (i == 1)				 p_d[l] += (-2 * p0_d[l] + p0_d[l + 1] + dx1_eq_0_forward(l, p0_d))/(hx_d*hx_d);
			//else if (i == nx_d - 1)  p_d[l] += (-2 * p0_d[l] + p0_d[l - 1] + dx1_eq_0_back(l, p0_d)) / (hx_d*hx_d);
			//else					 p_d[l] += dx2_center(l, p0_d);

			//if (j == 1)				 p_d[l] += (-2 * p0_d[l] + p0_d[l + offset] + (4.0*p0_d[l + offset] - p0_d[l + offset * 2]) / 3.0 - coef*(bettaT_d*(T0_d[l] - Temp0_d) + bettaC_d *(C0_d[l] - Con0_d)) * (2.0 * hy_d) / 3.0) / (hy_d*hy_d);
			//else if (j == ny_d - 1)  p_d[l] += (-2 * p0_d[l] + p0_d[l - offset] + (4.0*p0_d[l - offset] - p0_d[l - offset * 2]) / 3.0 + coef*(bettaT_d*(T0_d[l] - Temp0_d) + bettaC_d *(C0_d[l] - Con0_d)) * (2.0 * hy_d) / 3.0) / (hy_d*hy_d);
			//else					 p_d[l] += dy2_center(l, p0_d);

			//if (i == 1)					dx2_forward(l, p0_d);
			//else if (i == nx_d - 1)  dx2_back(l, p0_d);
			//else					  dx2_center(l, p0_d);

			//if (j == 1)				dy2_up(l, p0_d);
			//else if (j == ny_d - 1) dy2_down(l, p0_d);
			//else					 dy2_center(l, p0_d);

			p_d[l] *= tau_p_d;
			p_d[l] += p0_d[l];
		}


		else if (j == 0 && (i > 0 && i < nx_d))		p_d[l] = (4.0*p0_d[l + offset] - p0_d[l + offset * 2]) / 3.0 -coef*(bettaT_d*(T0_d[l] - Temp0_d) + bettaC_d *(C0_d[l] - Con0_d)) * (2.0 * hy_d) / 3.0;
		else if (j == ny_d && (i > 0 && i < nx_d))		p_d[l] = (4.0*p0_d[l - offset] - p0_d[l - offset * 2]) / 3.0 +coef*(bettaT_d*(T0_d[l] - Temp0_d) + bettaC_d *(C0_d[l] - Con0_d)) * (2.0 * hy_d) / 3.0;
		//else if (j == 0 && (i > 0 && i < nx_d))		p_d[l] = (p0_d[l + offset]) - coef*(bettaT_d*(T0_d[l] - Temp0_d) + bettaC_d *(C0_d[l] - Con0_d)) * (hy_d) ;
		//else if (j == ny_d && (i > 0 && i < nx_d))		p_d[l] = (p0_d[l - offset])  + coef*(bettaT_d*(T0_d[l] - Temp0_d) + bettaC_d *(C0_d[l] - Con0_d)) * (hy_d) ;

		//else if (j == 0 && (i > 0 && i < nx_d))		p_d[l] = (3.0*p0_d[l + offset] - 3.0 / 2.0* p0_d[l + offset * 2] + 1.0 / 3.0*p0_d[l + offset * 3]) * 6.0 / 11.0 - coef*(bettaT_d*(T0_d[l] - Temp0_d) + bettaC_d *(C0_d[l] - Con0_d)) * (6.0 / 11.0 * hy_d);
		//else if (j == ny_d && (i > 0 && i < nx_d))		p_d[l] = (3.0*p0_d[l - offset] - 3.0 / 2.0* p0_d[l - offset * 2] + 1.0 / 3.0*p0_d[l - offset * 3]) * 6.0 / 11.0 + coef*(bettaT_d*(T0_d[l] - Temp0_d) + bettaC_d *(C0_d[l] - Con0_d)) * (6.0 / 11.0 * hy_d);



		else if (i == 0 && (j > 0 && j < ny_d))		p_d[l] =  dx1_eq_0_forward(l, p0_d);
		else if (i == nx_d && (j > 0 && j < ny_d))		p_d[l] = dx1_eq_0_back(l, p0_d);


		else if (i <= nx_d && j <= ny_d && l < n_d) {
			int ii = i + 1 - i / nx_d - ceil(i / (nx_d + 1.0));
			int jj = j + 1 - j / ny_d - ceil(j / (ny_d + 1.0));
			p_d[l] = p0_d[ii + offset*jj];
		}
	}
}
__global__ void Velocity_Darcy(double *vx_d, double *vy_d, double *vz_d, double *p0_d, double *T0_d, double *C0_d) {
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int l = i + offset*j;

	/*
	closed
	*/
	double coef = Ra_d*grav_d;
	double coef2 = K_d / nu_d;
	if (border_type == 0 && i <= nx_d && j <= ny_d && l < n_d)
	{
		if (i > 0 && i < nx_d && j > 0 && j < ny_d)
		{
			vy_d[l] = (-dy1_center(l, p0_d) / rho_d + coef*(bettaT_d*(T0_d[l] - Temp0_d) + bettaC_d*(C0_d[l] - Con0_d)))* coef2;
			vx_d[l] = (-dx1_center(l, p0_d) / rho_d) * coef2;
		}


		else if (j == 0 && (i > 0 && i < nx_d))		vy_d[l] = 0;
		else if (j == ny_d && (i > 0 && i < nx_d))		vy_d[l] = 0;
		else if (i == 0 && (j > 0 && j < ny_d))		vx_d[l] = 0;
		else if (i == nx_d && (j > 0 && j < ny_d))		vx_d[l] = 0;

		else if (i <= nx_d && j <= ny_d && l < n_d) {
			vx_d[l] = vy_d[l] = 0;
		}
	}
}
#endif // Stream

#endif // level2


__global__ void reduction00(double *data, unsigned int n, double* reduced) {
	extern __shared__ double shared[];


	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

	shared[tid] = (i < n) ? abs(data[i]) : 0;

	if (i + blockDim.x < n)	shared[tid] += abs(data[i + blockDim.x]);


	__syncthreads();


	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>32; s >>= 1)
	{
		if (tid < s)
		{
			shared[tid] += shared[tid + s];
		}

		__syncthreads();
	}


	if (tid < 32)
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockDim.x >= 64) shared[tid] += shared[tid + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize / 2; offset > 0; offset /= 2)
		{
			shared[tid] += __shfl_down((float)shared[tid], offset);
		}
	}



	if (tid == 0) {
		reduced[blockIdx.x] = shared[0];
	}


}
__global__ void reduction0(double *data, unsigned int n, double* reduced) {
	extern __shared__ double shared[];


	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < n) {
		shared[tid] = abs(data[i]);
	}
	else
	{
		shared[tid] = 0.0;
	}

	__syncthreads();


	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s)
		{
			shared[tid] += shared[tid + s];
		}

		__syncthreads();
	}

	if (tid == 0) {
		reduced[blockIdx.x] = shared[0];
	}


}
__global__ void reduction(double *data, unsigned int n, double* reduced) {
	extern  __shared__  double shared[];

	unsigned int tid = threadIdx.x;
	//unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) {
		shared[tid] = abs(data[i]);
		//if (i + blockDim.x  < n) shared[tid] += abs(data[i + blockDim.x]);
	}
	else
	{
		shared[tid] = 0.0;
	}

	__syncthreads();

	if (blockDim.x >= 1024) {
		if (tid < 512) { shared[tid] += shared[tid + 512]; } __syncthreads();
	}
	if (blockDim.x >= 512) {
		if (tid < 256) { shared[tid] += shared[tid + 256]; } __syncthreads();
	}
	if (blockDim.x >= 256) {
		if (tid < 128) { shared[tid] += shared[tid + 128]; } __syncthreads();
	}
	if (blockDim.x >= 128) {
		if (tid < 64) { shared[tid] += shared[tid + 64]; } __syncthreads();
	}
	if (tid < 32)
	{
		if (blockDim.x >= 64) shared[tid] += shared[tid + 32];
		if (blockDim.x >= 32) shared[tid] += shared[tid + 16];
		if (blockDim.x >= 16) shared[tid] += shared[tid + 8];
		if (blockDim.x >= 8) shared[tid] += shared[tid + 4];
		if (blockDim.x >= 4) shared[tid] += shared[tid + 2];
		if (blockDim.x >= 2) shared[tid] += shared[tid + 1];
	}




	if (tid == 0) {
		reduced[blockIdx.x] = shared[0];
		//if (blockDim.x==1) *last = shared[0];
	}


}


__global__ void swap_one(double* f_old, double* f_new) {
	unsigned int l = blockIdx.x*blockDim.x + threadIdx.x;
	if (l < n_d)	f_old[l] = f_new[l];
}
__global__ void swap_5(double* f1_old, double* f1_new, double* f2_old, double* f2_new, double* f3_old, double* f3_new, double* f4_old, double* f4_new, double* f5_old, double* f5_new) {
	unsigned int l = blockIdx.x*blockDim.x + threadIdx.x;
	if (l < n_d)
	{
		f1_old[l] = f1_new[l];
		f2_old[l] = f2_new[l];
		f3_old[l] = f3_new[l];
		f4_old[l] = f4_new[l];
		f5_old[l] = f5_new[l];
	}
}



__global__ void pressure_norm1(double* T_d, int i, int j) {
	dp = T_d[i + j * offset];
	//dp = - psi2_d*1.0 + T_d[nx_d / 2 + nz_d / 2 * offset2];
}
__global__ void pressure_norm2(double* T_d) {
	unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y*blockDim.y;
	unsigned int l = i + offset*j;

	if (l < n_d) {
		T_d[l] = T_d[l] - dp;
	}
}
__global__ void pressure_point(double* p, int i, int j, double val) {
	p[i + offset*j] = val;
}

#ifdef level3
#define min3(a,b,c) min(a,min(b,c))
#endif
#ifdef level2
#define min3(a,b,c) min(a,b)
#endif 


int grid_build2(double **h_array, double **coord, unsigned int &N, double &h, double L, double dL, double hmin, double mult = 1) {
	if (N < 10) {
		Log << "too small grid " << endl;
		return 1;
	}

	double dh;
	int n = (int)ceil(dL / h); dL = h*n;
	int m = (int)ceil(dL / hmin);
	double Lm;
	int Nm;

	int num = (int)(mult*n);
	Log << "n = " << n << " m = " << m << " num = " << num << endl;


	//dh = 2*(dL - h - (N+1)*hmin)/(N*(N+1));
	while (true) {
		//Log << "num = " << num << " dh = " << dh << endl;
		dh = 2.0 * (num*h - dL) / (num*(num - 1));
		if (h - (num - 1)*dh > hmin) break;
		num = num - 1;
	}

	//double sum = 0;
	//for (int i = 0; i < num; i++) {
	//	Log << i << " " << h - (num - 1 - i)*dh << " " << (h - (num - 1 - i)*dh) / h * 100  <<"%"<< endl;
	//	sum += h - (num - 1 - i)*dh;
	//}

	Lm = L - 2.0*dL;
	Nm = (int)ceil(Lm / h);
	h = Lm / Nm;

	N = 2 * num + Nm;
	*h_array = (double*)malloc(sizeof(double)*N);
	*coord = (double*)malloc(sizeof(double)*(N+1));
	if (*h_array == NULL) Log << "WTF?" << endl;
	Log << "dh: " << dh << " h: " << h << " hmin: " << hmin << endl;
	Log << "dL: " << dL << " Lm: " << Lm << " L: " << L << endl;
	Log << "n: " << num << " Nm: " << Nm << " N: " << N << endl;

	for (int i = 0; i < N; i++) {
		if (i < num)
			(*h_array)[i] = h - (num - 1 - i)*dh;
		if (i >= num && i < Nm + num)
			(*h_array)[i] = h;
		if (i >= Nm + num)
			(*h_array)[i] = h - (-(Nm + num) + i)*dh;
	}
	
	double s = 0;
	(*coord)[0] = 0;
	for (int i = 0; i < N; i++) {
		s += (*h_array)[i];
		(*coord)[i + 1] = s;
		//std::cout << i << " " << (*h_array)[i] << " " << s << endl;
	}
	if ((float)s == (float)L) Log << "properly constructed" << endl;
	return 0;
}

int grid_build(double **h_array, unsigned int &N, double &h, double L, double dL, double dh, int reduce = 0) {
	double Lm;
	int n, Nm;

	if (N < 10) {
		Log << "too small grid " << endl;
		return 1;
	}

	if (reduce == 0)
		while (true) {
			n = (int)ceil(-0.5 + 0.5 * sqrt(1.0 + 8.0*dL / dh));
			dh = 2.0*dL / (n*n + n);

			if (dh*n > h) {
				dL = dL * 0.9;  //or another wayы: n = n -1
				Log << "dL is reduced = " << dL << endl;
			}
			else break;
		}
	else if (reduce == 1) {
		Log << "uniform grid" << endl;
		*h_array = (double*)malloc(sizeof(double)*N);
		for (int i = 0; i < N; i++) {
				(*h_array)[i] = h;
		}
		return 0;
	}
	else {
		Log << "reduce: " << reduce << endl;
		n = (int)ceil(-0.5 + 0.5 * sqrt(1.0 + 8.0*dL / h));
		n = n * reduce;
		dh = 2.0*dL / (n*n + n);
	}

	Lm = L - 2.0*dL;
	Nm = (int)ceil(Lm / h);
	h = Lm / Nm;

	N = 2 * n + Nm;
	*h_array = (double*)malloc(sizeof(double)*N);
	if (*h_array == NULL) Log << "WTF?" << endl;

	Log << "edge h: " << dh << " middle h: " << h << endl;
	Log << "edge L: " << dL << " middle L: " << Lm << " the whole L: " << L << endl;
	Log << "edge nodes: " << n << " middle nodes: " << Nm << " all nodes: " << N << endl;


	for (int i = 0; i < N; i++) {
		if (i < n)
			(*h_array)[i] = dh*(i + 1);
		if (i >= n && i < Nm + n)
			(*h_array)[i] = h;
		if (i >= Nm + n)
			(*h_array)[i] = dh*(-i + N);
	}
	double s = 0;
	for (int i = 0; i < N; i++) {
		s += (*h_array)[i];
		//cout << i << " " << (*h_array)[i] << " " << s << endl;
	}
	if ((float)s == (float)L) Log << "properly constructed" << endl;
	return 0;
}
void grid_near_border(int switch_on ) {
	if (switch_on == 1) {
		//Log << endl << "nx grid near border:" << endl; grid_build(&HX, nx, hx, Lx, 0.1*Lx, 0.01*Lx, 1);
		//Log << endl << "ny grid near border:" << endl; grid_build(&HY, ny, hy, Ly, 0.1*Ly, 0.01*Ly, 0);
		//Log << endl << "nz grid near border:" << endl; grid_build(&HZ, nz, hz, Lz, 0.1*Lz, 0.01*Lz, 1);
		Log << endl << "nx grid near border:" << endl; grid_build2(&HX, &XX, nx, hx, Lx, 0.1*Lx, hx / 3, 1);
		Log << endl << "ny grid near border:" << endl; grid_build2(&HY, &YY, ny, hy, Ly, 0.1*Ly, hy / 3, 5);
		Log << endl << "nz grid near border:" << endl; grid_build2(&HZ, &ZZ, nz, hz, Lz, 0.1*Lz, hz / 3, 1);
		cudaMalloc((void**)&DX, sizeof(double)*nx);
		cudaMalloc((void**)&DY, sizeof(double)*ny);
		cudaMalloc((void**)&DZ, sizeof(double)*nz);

		cudaMemcpy(DX, HX, nx * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(DY, HY, ny * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(DZ, HZ, nz * sizeof(double), cudaMemcpyHostToDevice);

		cudaMemcpyToSymbol(dx, &DX, sizeof(double*));
		cudaMemcpyToSymbol(dy, &DY, sizeof(double*));
		cudaMemcpyToSymbol(dz, &DZ, sizeof(double*));
	}
}




struct ReadingFile
{
private:
	ifstream read;
	string str, substr, buffer;
	string file_name;
	stringstream ss;
	istringstream iss;
	ostringstream oss;
	int stat, pos;

public:
	ReadingFile(string name)
	{
		file_name = name;
		open_file(file_name);
		stat = 0;
	}

	void open_file(string file_name) {
		read.open(file_name.c_str());
		if (read.good()) {
			cout << endl << "the parameter file \"" << file_name << "\" has been read " << endl << endl;
			Log << endl << "the parameter file \"" << file_name << "\" has been read " << endl << endl;
			oss << read.rdbuf();
			buffer = oss.str();
			iss.str(buffer);
		}
		else {
			cout << "the parameter file has been not found, default parameters will be initialized " << endl;
			Log << "the parameter file has been not found, default parameters will be initialized " << endl;
			buffer = "";
			iss.str(buffer);
		}
	}


	template <typename T>
	void reading(T &var, string parameter_name, T def_var, T min = 0, T max = 0) {
		stat = 0;
		Log << parameter_name << "= ";
		transform(parameter_name.begin(), parameter_name.end(), parameter_name.begin(), ::tolower);
		iss.clear();
		iss.seekg(0);

		while (getline(iss, str))
		{
			//substr.clear();
			ss.str("");	ss.clear();	ss << str;	ss >> substr;
			transform(substr.begin(), substr.end(), substr.begin(), ::tolower);
			if (substr == parameter_name) {
				pos = (int)ss.tellg();
				while (ss >> substr) {
					if (substr == "=")
					{
						ss >> var;
						stat = 1;					
						break;
					}
				}

				if (stat == 0) {
					ss.clear();
					ss.seekg(pos);
					ss >> var;
				}
				break;
			}
		}
		if (iss.fail())
		{
			var = def_var;
		}

		if (min != max && (min + max) != 0) {
			if (var > max || var < min)
			{
				cout << "Warning: \"" + parameter_name + "\" should not be within this range" << endl;
				var = def_var;
			}
		}
		Log << var;
		if (stat == 0) 
			Log << " (default)";
		Log << endl;
	}

	void reading_string(string &var, string parameter_name, string def_var) {
		stat = 0;
		Log << parameter_name << "= ";
		transform(parameter_name.begin(), parameter_name.end(), parameter_name.begin(), ::tolower);
		iss.clear();
		iss.seekg(0);

		while (getline(iss, str))
		{
			//substr.clear();
			ss.str("");	ss.clear();	ss << str;	ss >> substr;
			transform(substr.begin(), substr.end(), substr.begin(), ::tolower);
			if (substr == parameter_name) {
				pos = (int)ss.tellg();
				while (ss >> substr) {
					if (substr == "=")
					{
						ss >> var;
						stat = 1;
						break;
					}
				}

				if (stat == 0) {
					ss.clear();
					ss.seekg(pos);
					ss >> var;
				}
				break;
			}
		}
		if (iss.fail())
		{
			var = def_var;
		}
		Log << var;
		if (stat == 0)
			Log << " (default)";
		Log << endl;
	}


};
void setParameters(int argc, char **argv) {
	string file_name = "inp.dat";
	if (argc == 2) file_name = argv[1];
	ReadingFile File(file_name);

	//porB = Le = Ra = 1; psi = -1;
	File.reading<double>(Ra, "Ra", 1);
	File.reading<double>(Le, "Le", 1);
	File.reading<double>(psi, "psi", -1);
	File.reading<double>(porB, "porB", 1);


	File.reading<double>(K, "K", 5.66e-10);
	File.reading<double>(por, "por", 0.48);
	File.reading<double>(porB, "porB", por);
	File.reading<double>(bettaT, "bettaT", 7.86e-4);
	File.reading<double>(bettaC, "bettaC", -0.212);
	File.reading<double>(rho, "rho", 935.17);
	File.reading<double>(rho_por, "rho_por", 1.0);
	File.reading<double>(Cp, "Cp", 1.0);
	File.reading<double>(Cp_por, "Cp_por", 1.0);
	File.reading<double>(nu,"nu", 2.716e-6);
	File.reading<double>(grav, "grav", 9.81);
	File.reading<double>(D, "D", (4.323e-10)); D = D / 2.30;
	File.reading<double>(Dt_, "Dt_", (1.37e-12)); Dt_ = Dt_ / 2.27;
	File.reading<double>(dT, "incrT", 0);

	File.reading<double>(Con0, "Con0", 0.6088);
	File.reading<double>(Dt, "Dt", Con0*(1.0 - Con0)*Dt_);

	File.reading<double>(rhoCpF, "rhoCpF", 3.28e+6);
	File.reading<double>(rhoCpP, "rhoCpP", 2.51e+6);
	File.reading<double>(kappa_p, "kappa_p", 0.731);
	File.reading<double>(a, "a", kappa_p / rhoCpF);
	File.reading<double>(chi, "chi", kappa_p / rhoCpP);
	File.reading<double>(St_, "St_", Dt_/D);
	File.reading<double>(St, "St", Con0*(1.0 - Con0)*St_);

	File.reading<double>(T_hot, "T_hot", 25);
	File.reading<double>(T_cold, "T_cold", 15);
	File.reading<double>(Temp0, "Temp0", T_cold);
	File.reading<double>(deltaTemp, "deltaTemp", T_hot - T_cold);
	File.reading<double>(tau, "tau", 1e-2);

	File.reading<double>(Lz, "Lz", 0.006);
	File.reading<double>(Ly, "Ly", 0.3);
	File.reading<double>(Lx, "Lx", 0.1);

	File.reading<unsigned int>(nz, "nz", 30);
	File.reading<unsigned int>(ny, "ny", 30);
	File.reading<unsigned int>(nx, "nx", 30);

	File.reading<int>(dimensionless, "dimensionless", 0, 0, 1);
	File.reading<int>(read_recovery, "read_recovery", 0);
	File.reading<double>(StopIncr, "StopIncr", 500);
	File.reading<double>(minimumTime, "minumumTIme", 10000.0);
	File.reading<int>(grid_border, "grid_border", 0, 0, 1);

	//bettaC = 0;
	//bettaT = -1;
	//deltaTemp = 10 / (K*grav*bettaT*min3(Lx, Ly, Lz))*nu*(rhoCpP / rhoCpF * chi); 	T_hot = T_cold + deltaTemp;
	cout << "Ra= " << K*grav*bettaT*min3(Lx, Ly, Lz)*deltaTemp / nu / (a) << endl << endl;

	coefT = deltaTemp; shiftT = Temp0;
	coefC = bettaT / bettaC * deltaTemp; shiftC = Con0;

	

	if (dimensionless) { //if dimensionaless

		Le = a / D;
		porB = por * rhoCpF / rhoCpP;
		psi = - bettaC / bettaT * Dt / D;
		
		L = min3(Lx, Ly, Lz);
		Lx /= L; Ly /= L; Lz /= L;


		File.reading <double>(Ra, "Ra", Ra = K*grav*bettaT*L*deltaTemp / nu / (a));
		bettaT = bettaC = a = chi = rhoCpF = rhoCpP = nu = grav = D = St = Dt = Dt_ = deltaTemp = T_hot = rho = rho_por = Cp = Cp_por = L = K = por  = 1;
		Temp0 = Con0 = T_cold = 0;

		tau = tau * 1;
		//Ra = 100;
		//psi = 0;
		//tau = tau * chi / (L * L);

	}
	//bettaC = 0;

	File.reading<double>(hx, "hx", Lx / nx);
	File.reading<double>(hy, "hy", Ly / ny);
	File.reading<double>(hz, "hz", Lz / nz);
	//hx = Lx / nx; hy = Ly / ny;	hz = Lz / nz;
	//pause

	Log << " " << endl;
	File.reading<unsigned int>(tt, "tt", 10000);//tt = round(1.0 / tau);
	File.reading<double>(eps0, "eps0", 1e-5);
	border = 0; 	//0 is for the closed cavity, 1 is for the periodic one
	pi = 3.1415926535897932384626433832795;
	printParameters();
	
}



void auxiliarySetup() {

#ifdef __linux__
	system("mkdir -p fields/");
#endif

#ifdef _WIN32
	CreateDirectoryA("fields", NULL);
#endif
	m = 0.0; psiav = 0.0; iter = 0; Ek = 0; Ek_old = 0; timeq = 0.0;
	tau_p = 0.1*pow(min3(hx, hy, hz), 2);
	off /*offset*/ = nx + 1;
	off2 /*offset2*/ = (nx + 1)* (ny + 1);
#ifdef level2
	off2 = 0;
	nz = 0;
#endif 
	size_l = (nx + 1) * (ny + 1) * (nz + 1); //Number of all nodes/elements 
	size_b = size_l * sizeof(double); //sizeof(double) = 8 bytes

	Log << "final sizes:" << endl;
	Log << "nx: " << nx << " ny: " << ny << " nz: " << nz << endl;
	Log << "offset1: " << off << " offset2:" << off2 << endl;
	Log << "size_l: " << size_l << " size_b: " << size_b << endl;

	cout << "approximate amount of memory: " << size_l * sizeof(double) * 12 / 1024 / 1024 << " MB" << endl;
	Log << "approximate amount of memory: " << size_l * sizeof(double) * 12 / 1024 / 1024 << " MB" << endl;
	if (size_l <= 1024 || size_l >= 1024 * 1024 * 1024) {
		cout << "data is too small or too large" << endl;
		Log << "data is too small or too large" << endl;
	}
	



#ifdef level2
	thread_x_d = 32;
	thread_y_d = 32;
	Log << "2D" << endl;
#endif 

#ifdef level3
	thread_x_d = 8;
	thread_y_d = 8;
	thread_z_d = 8;
	Log << "3D" << endl;
#endif // level3



}
void initial_level(bool copy_to_GPU = false) {
	double pressure_slope = -0.5*bettaT*grav*deltaTemp*rho; //36.026


#ifdef level3
	for (int i = 0; i <= nx; i++) {
		for (int j = 0; j <= ny; j++) {
			for (int k = 0; k <= nz; k++) {
				//T_h[i + j*off + k*off2] = T_hot - (T_hot - T_cold) / Ly *(j*hy);

				T_h[i + j*off + k*off2] = T_hot - (T_hot - T_cold) / Lz *(k*hz);
				C_h[i + j*off + k*off2] = Con0;
				p_h[i + j*off + k*off2] = pressure_slope*k*hz;

			}
		}
	}
#endif // level3
#ifdef level2
	

	for (int i = 0; i <= nx; i++) {
		for (int j = 0; j <= ny; j++) {
			T_h[i + j*off] = T_hot - (T_hot - T_cold) / Lx *(i*hx);
#ifdef BorderMesh
			T_h[i + j*off] = T_hot - (T_hot - T_cold) / Lx *(XX[i]);
#endif // BorderMesh

			C_h[i + j*off] = Con0;
			//C_h[i + j*off] = 0.6153 - 0.02475*(j*hy);
#ifndef Stream
			p_h[i + j*off] = pressure_slope*j*hy;  //- 35.89967633/2*Ly;
#endif // !STREAM


			
		}
	}
#endif 
	//T_h[nx / 2 + ny / 2 * off + nz / 2 * off2] = T_hot;
	if (copy_to_GPU) {
		cudaMemcpy(C0_d, C_h, size_b, cudaMemcpyHostToDevice);		cudaMemcpy(C_d, C_h, size_b, cudaMemcpyHostToDevice);
		cudaMemcpy(T0_d, T_h, size_b, cudaMemcpyHostToDevice);		cudaMemcpy(T_d, T_h, size_b, cudaMemcpyHostToDevice);
		cudaMemcpy(p0_d, p_h, size_b, cudaMemcpyHostToDevice);		cudaMemcpy(p_d, p_h, size_b, cudaMemcpyHostToDevice);
	}
}



void allocationCPU() {

	auto alloc = [](double **f)
	{
		*f = (double*)malloc(size_b);
		for (int i = 0; i < size_l; i++) 	(*f)[i] = 0.0;
	};
	alloc(&p_h);
	alloc(&vx_h);
	alloc(&vy_h);
	alloc(&vz_h);
	alloc(&T_h);
	alloc(&C_h);
	alloc(&test_h);
}

void allocationGPU_test() {
	auto alloc = [](double **d, double *h)
	{
		cudaMalloc((void**)d, size_b);
		cudaMemcpy((*d), h, size_b, cudaMemcpyHostToDevice);
	};

	alloc(&p_d, p_h);	alloc(&p0_d, p_h);
	alloc(&ux_d, vx_h);	alloc(&uy_d, vy_h);	alloc(&uz_d, vz_h);
	alloc(&vx_d, vx_h);	alloc(&vy_d, vy_h);	alloc(&vz_d, vz_h);
	alloc(&C_d, C_h);	alloc(&C0_d, C_h);
	alloc(&T_d, T_h);	alloc(&T0_d, T_h);
	alloc(&test_d, test_h);

	(N_reduce != 1) ? cudaMalloc((void**)&psiav_array, sizeof(double)*N_p[1]) : cudaMalloc((void**)&psiav_array, sizeof(double));
	cudaMalloc((void**)&psiav_d, sizeof(double));
	arr[0] = p_d;
	for (int i = 1; i <= N_reduce; i++)
		arr[i] = psiav_array;
}

void allocationGPU() {
	//allocating memory for arrays on GPU


	cudaMalloc((void**)&p_d, size_b); 
	cudaMalloc((void**)&p0_d, size_b);
	cudaMalloc((void**)&ux_d, size_b);	cudaMalloc((void**)&uy_d, size_b); 	cudaMalloc((void**)&uz_d, size_b);
	cudaMalloc((void**)&vx_d, size_b);	cudaMalloc((void**)&vy_d, size_b); 	cudaMalloc((void**)&vz_d, size_b);
	cudaMalloc((void**)&C_d, size_b);  cudaMalloc((void**)&C0_d, size_b);
	cudaMalloc((void**)&T_d, size_b);  cudaMalloc((void**)&T0_d, size_b);
	cudaMalloc((void**)&test_d, size_b);
	(N_reduce != 1) ? cudaMalloc((void**)&psiav_array, sizeof(double)*N_p[1]) : cudaMalloc((void**)&psiav_array, sizeof(double));
	cudaMalloc((void**)&psiav_d, sizeof(double));

	arr[0] = p_d;
	for (int i = 1; i <= N_reduce; i++)
		arr[i] = psiav_array;


	cudaMemcpy(p0_d, p_h, size_b, cudaMemcpyHostToDevice); 	
	cudaMemcpy(p_d, p_h, size_b, cudaMemcpyHostToDevice);
	cudaMemcpy(ux_d, vx_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(uy_d, vy_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(uz_d, vz_h, size_b, cudaMemcpyHostToDevice);
	cudaMemcpy(vx_d, vx_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(vy_d, vy_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(vz_d, vz_h, size_b, cudaMemcpyHostToDevice);
	cudaMemcpy(C0_d, C_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(C_d, C_h, size_b, cudaMemcpyHostToDevice);
	cudaMemcpy(T0_d, T_h, size_b, cudaMemcpyHostToDevice); 	cudaMemcpy(T_d, T_h, size_b, cudaMemcpyHostToDevice);

	cudaMemcpy(test_d, test_h, size_b, cudaMemcpyHostToDevice);

}

void Poisson_setup() {
	//setting for the reduction procedure 
	N_reduce = 0, thread_p = 1024;


	unsigned int GN = size_l;
	while (true)
	{
		N_reduce++;
		GN = ceil(GN / (thread_p + 0.0));
		if (GN == 1)  break;
	}
	GN = size_l;
	std::cout << "the number of reduction = " << N_reduce << endl;
	Grid_p = new unsigned long long int[N_reduce];
	N_p = new unsigned long long int[N_reduce];
	for (int i = 0; i < N_reduce; i++)
		Grid_p[i] = GN = ceil(GN / (thread_p + 0.0));
	N_p[0] = size_l;
	for (int i = 1; i < N_reduce; i++)
		N_p[i] = Grid_p[i - 1];
	int last_reduce = pow(2, ceil(log2(N_p[N_reduce - 1] + 0.0))); //last_reduce = pow(2, ceil(log2(size_l / 1024)));
	std::cout << "last reduction = " << last_reduce << endl;
	(N_reduce != 1) ? std::cout << "sub array for the Poisson solver = " << N_p[1] << endl :
		std::cout << "it shouldn't be here" << endl;
}

void reading_recover(int read_fields = 1) {




	ifstream read("all.txt");
	(read.good() == true && read_fields > 0) ? cout << "a file is opened to continue" << endl : cout << " a file is not found, I'l do it from the start" << endl;
	if (read.good() == false) read_fields = 0;
	
	Log << "Reading recovery: " << read_fields << endl;
	if (read_fields == 0) {
		Ra_tab.open("Ra.dat");
		integrals.open("integrals.dat");
		Ra_tab << "Heat, Ek, Vmax, dCmax, Ctotal, time(min), t" << endl;


		//начальные условия
		for (int i = 0; i <= nx; i++) {
			for (int j = 0; j <= ny; j++) {
				for (int k = 0; k <= nz; k++) {
					vx_h[i + off*j + off2*k] = 0.0;
					vy_h[i + off*j + off2*k] = 0.0;
					vz_h[i + off*j + off2*k] = 0.0;
					p_h[i + off*j + off2*k] = 0.0;
					T_h[i + off*j + off2*k] = 0.0;
					C_h[i + off*j + off2*k] = 0.0;
				}
			}
		}
		initial_level();
		//vy_h[nx / 2 + off * ny / 2 + off2 * nz / 2] = 0.1;

	}
	else if (read_fields == 1)
	{
		Ra_tab.open("Ra.dat", std::ofstream::app);
		integrals.open("integrals.dat", std::ofstream::app);


		string str;
		string substr;
		stringstream ss;
		getline(read, str); //head
		ss.str("");	ss.clear();	ss << str;	
		
		while (ss >> substr) {
			if (substr == "time=") 	ss >> timeq;
			if (substr == "iter=") 	ss >> iter;
		}
		
//		ss << str; ss >> substr; ss >> substr; iter = atoi(substr.c_str()); //time
//		ss >> substr; ss >> substr; Ra = atof(substr.c_str()); Ra -= dRa; //Ra
		int l;
		for (int i = 0; i <= nx; i++) {
			for (int j = 0; j <= ny; j++) {
				for (int k = 0; k <= nz; k++) {
					l = i + off*j + off2*k;
					ss.str(""); ss.clear(); getline(read, str); ss << str;
					ss >> substr; ss >> substr; ss >> substr; //skip reading x,y,z
					ss >> substr; vx_h[l] = atof(substr.c_str());
					ss >> substr; vy_h[l] = atof(substr.c_str());
					ss >> substr; vz_h[l] = atof(substr.c_str());
					ss >> substr; p_h[l] = atof(substr.c_str());
					ss >> substr; T_h[l] = atof(substr.c_str());
					ss >> substr; C_h[l] = atof(substr.c_str());
				}
			}
		}

		cout << "continue from t= " << timeq << " (" << iter << " iter)" << endl;
	}
	else if (read_fields == 2) {
		Ra_tab.open("Ra.dat");
		integrals.open("integrals.dat");
		Ra_tab << "Heat, Ek, Vmax, dCmax, Ctotal, time(min), t" << endl;
		string str;
		string substr;
		stringstream ss;
		getline(read, str); //head
		int l;
		for (int i = 0; i <= nx; i++) {
			for (int j = 0; j <= ny; j++) {
				for (int k = 0; k <= nz; k++) {
					l = i + off*j + off2*k;
					ss.str(""); ss.clear(); getline(read, str); ss << str;
					ss >> substr; ss >> substr; ss >> substr; //skip reading x,y,z
					ss >> substr; vx_h[l] = atof(substr.c_str());
					ss >> substr; vy_h[l] = atof(substr.c_str());
					ss >> substr; vz_h[l] = atof(substr.c_str());
					ss >> substr; p_h[l] = atof(substr.c_str());
					ss >> substr; T_h[l] = atof(substr.c_str());
					ss >> substr; C_h[l] = atof(substr.c_str());
				}
			}
		}
	}
	else if (read_fields == 3)
	{
		Ra_tab.open("Ra.dat");
		integrals.open("integrals.dat");
		Ra_tab << "Heat, Ek, Vmax, dCmax, Ctotal, time(min), t" << endl;
		string str;
		string substr;
		stringstream ss;
		getline(read, str); //head
		int l;
		for (int i = 0; i <= nx; i++) {
			for (int j = 0; j <= ny; j++) {
				for (int k = 0; k <= nz; k++) {
					l = i + off*j + off2*k;
					ss.str(""); ss.clear(); getline(read, str); ss << str;
					ss >> substr; ss >> substr; ss >> substr; //skip reading x,y,z
					ss >> substr; vx_h[l] = 0;
					ss >> substr; vy_h[l] = 0;
					ss >> substr; vz_h[l] = 0;
					ss >> substr; p_h[l] = atof(substr.c_str());
					ss >> substr; T_h[l] = 0;
					ss >> substr; C_h[l] = 0;
				}
			}
		}
		initial_level();
	}
	read.close();

}


//#define constCopy(var) cudaMemcpyToSymbol(var_d, &var, sizeof(double), 0, cudaMemcpyHostToDevice);
void copyConstantsGPU() {
	//copying some constant parameters to the fast constant memory

	cudaMemcpyToSymbol(hx_d, &hx, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(hy_d, &hy, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(hz_d, &hz, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Lx_d, &Lx, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Ly_d, &Ly, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Lz_d, &Lz, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(nx_d, &nx, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(ny_d, &ny, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(nz_d, &nz, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(n_d, &size_l, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(tau_d, &tau, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(tau_p_d, &tau_p, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(offset, &off, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(offset2, &off2, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(border_type, &border, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(K_d, &K, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(bettaT_d, &bettaT, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(bettaC_d, &bettaC, sizeof(double), 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(chi_d, &chi, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(nu_d, &nu, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(grav_d, &grav, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(por_d, &por, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(D_d, &D, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Dt_d, &Dt, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(St_d, &St, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Con0_d, &Con0, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(T_hot_d, &T_hot, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(T_cold_d, &T_cold, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Temp0_d, &Temp0, sizeof(double), 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(rho_d, &rho, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(rho_por_d, &rho_por, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Cp_d, &Cp, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Cp_por_d, &Cp_por, sizeof(double), 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(rhoCpF_d, &rhoCpF, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(rhoCpP_d, &rhoCpP, sizeof(double), 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(Ra_d, &Ra, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Le_d, &Le, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(psi_d, &psi, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(porB_d, &porB, sizeof(double), 0, cudaMemcpyHostToDevice);


}

void write_fields(int time, int fields, int Par, string name, int val_sec = 0, string section = "") {

	string folder = "";

	if (fields == 1) {
#ifdef __linux__
		folder = "fields/";
#endif

#ifdef _WIN32
		folder = "fields\\";
#endif
	}


	int i0, j0, k0, nxQ, nyQ, nzQ;
	i0 = 0;
	j0 = 0;
	k0 = 0;

	nxQ = nx;
	nyQ = ny;
	nzQ = nz;

	if (section == "nx") {
		i0 = val_sec;
		nxQ = val_sec;
	}
	if (section == "ny") {
		j0 = val_sec;
		nyQ = val_sec;
	}
	if (section == "nz") {
		k0 = val_sec;
		nzQ = val_sec;
	}


	stringstream ssTime, ssIncr; string str;
	ssTime.str(""); ssTime.clear();
	if (time == 1) 	ssTime << iter*tau;
	if (Par == 1)	ssIncr << Ra*deltaTemp;


	ofstream file((folder + ssTime.str() + name + ssIncr.str() + ".txt").c_str());


	int l;

	string table_head = "x, y, z, vx, vy, vz, p, T, C";
	file << table_head;
	if (time == 1) {
		file << " time= " << ssTime.str();
		file << " iter= " << iter;
	}
	file << endl;

	//all << setprecision(16) << fixed;

	

	for (int i = i0; i <= nxQ; i++) {
		for (int j = j0; j <= nyQ; j++) {
			for (int k = k0; k <= nzQ; k++) {
				l = i + off*j + off2*k;
				if (i == 0) {
					vy_h[l] = vy_h[l + 1];
					vz_h[l] = vz_h[l + 1];
				}
				if (i == nx) {
					vy_h[l] = vy_h[l - 1];
					vz_h[l] = vz_h[l - 1];
				}
				if (j == 0) {
					vx_h[l] = vx_h[l + off];
					vz_h[l] = vz_h[l + off];
				}
				if (j == ny) {
					vx_h[l] = vx_h[l - off];
					vz_h[l] = vz_h[l - off];
				}
				if (k == 0) {
					vy_h[l] = vy_h[l + off2];
					vx_h[l] = vx_h[l + off2];
				}
				if (k == nz) {
					vy_h[l] = vy_h[l - off2];
					vx_h[l] = vx_h[l - off2];
				}

				double x = i*hx, y = j*hy, z = k*hz;
#ifdef BorderMesh
				x = XX[i], y = YY[j], z = ZZ[k];
#endif
				file << x << " " << y << " " << z << " " << vx_h[l] << " " << vy_h[l] << " " << vz_h[l]  << " " << p_h[l] << " " << T_h[l] << " " << C_h[l] << endl;
			}
		}
	}
	
}

void write_fields_para(int time, int fields, string name, int val_sec = 0, string section = "") {

	string folder = "";

	if (fields == 1) {
#ifdef __linux__
		folder = "fields/";
#endif

#ifdef _WIN32
		folder = "fields\\";
#endif
	}


	int i0, j0, k0, nxQ, nyQ, nzQ;
	i0 = 0;
	j0 = 0;
	k0 = 0;

	nxQ = nx;
	nyQ = ny;
	nzQ = nz;

	if (section == "nx") {
		i0 = val_sec;
		nxQ = val_sec;
	}
	if (section == "ny") {
		j0 = val_sec;
		nyQ = val_sec;
	}
	if (section == "nz") {
		k0 = val_sec;
		nzQ = val_sec;
	}


	stringstream ssTime, ss2Ra; string str;
	ssTime.str(""); ssTime.clear();
	if (time == 1) {
		//ss << setprecision(15);
		ssTime << iter*tau;
	}



	ofstream file((folder + ssTime.str() + name + ".csv").c_str());


	int l;

	string table_head = "x, y, z, vx, vy, vz, p, T, C";
	file << table_head;
	if (time == 1) file << " time" << ssTime.str();
	file << endl;

	//all << setprecision(16) << fixed;



	for (int i = i0; i <= nxQ; i++) {
		for (int j = j0; j <= nyQ; j++) {
			for (int k = k0; k <= nzQ; k++) {
				l = i + off*j + off2*k;
				if (i == 0) {
					vy_h[l] = vy_h[l + 1];
					vz_h[l] = vz_h[l + 1];
				}
				if (i == nx) {
					vy_h[l] = vy_h[l - 1];
					vz_h[l] = vz_h[l - 1];
				}
				if (j == 0) {
					vx_h[l] = vx_h[l + off];
					vz_h[l] = vz_h[l + off];
				}
				if (j == ny) {
					vx_h[l] = vx_h[l - off];
					vz_h[l] = vz_h[l - off];
				}
				if (k == 0) {
					vy_h[l] = vy_h[l + off2];
					vx_h[l] = vx_h[l + off2];
				}
				if (k == nz) {
					vy_h[l] = vy_h[l - off2];
					vx_h[l] = vx_h[l - off2];
				}


				file << i << ", " << j << ", " << k << ", " << vx_h[l] << ", " << vy_h[l] << ", " << vz_h[l] << ", " << p_h[l] << ", " << T_h[l] << ", " << C_h[l] << endl;
			}
		}
	}

}


void backup() {


	ofstream all("all.txt");


	all << "#x, y, z, vx, vy, vz, C1, C2, T, p" << endl;
//	all << "#iter= " << iter << " Ra= " << Ra << endl;
	//all << setprecision(16) << fixed;
	int l;
	for (int i = 0; i <= nx; i++) {
		for (int j = 0; j <= ny; j++) {
			for (int k = 0; k <= nz; k++) {
				l = i + off*j + off2*k;
				all << i*hx << " " << j*hy << " " << k*hz << " " << vx_h[l] << " " << vy_h[l] << " " << vz_h[l] << " " << p_h[l] << " " << T_h[l] << " " << C_h[l] << endl;
			}
		}
	}
}

void copyFromGPU() {
	if (copied == 0) {
		cudaMemcpy(C_h, C_d, size_b, cudaMemcpyDeviceToHost);
		cudaMemcpy(T_h, T_d, size_b, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_h, p_d, size_b, cudaMemcpyDeviceToHost);
		cudaMemcpy(vx_h, vx_d, size_b, cudaMemcpyDeviceToHost);
		cudaMemcpy(vy_h, vy_d, size_b, cudaMemcpyDeviceToHost);
		cudaMemcpy(vz_h, vz_d, size_b, cudaMemcpyDeviceToHost);
		copied = 1;
	}
}

void write_test(int val_sec = 0, string section = "") {
	if (copied == 0) {
		cudaMemcpy(p_h, p_d, size_b, cudaMemcpyDeviceToHost);
		cudaMemcpy(vx_h, vx_d, size_b, cudaMemcpyDeviceToHost);
		cudaMemcpy(vy_h, vy_d, size_b, cudaMemcpyDeviceToHost);
		cudaMemcpy(vz_h, vz_d, size_b, cudaMemcpyDeviceToHost);

		cudaMemcpy(ux_h, ux_d, size_b, cudaMemcpyDeviceToHost);
		cudaMemcpy(uy_h, uy_d, size_b, cudaMemcpyDeviceToHost);
		cudaMemcpy(uz_h, uz_d, size_b, cudaMemcpyDeviceToHost);

		cudaMemcpy(T_h, T_d, size_b, cudaMemcpyDeviceToHost);
		cudaMemcpy(C_h, C_d, size_b, cudaMemcpyDeviceToHost);
		copied = 1;
	}



	string folder = "";

	int i0, j0, k0, nxQ, nyQ, nzQ;
	i0 = 0;
	j0 = 0;
	k0 = 0;

	nxQ = nx;
	nyQ = ny;
	nzQ = nz;

	if (section == "nx") {
		i0 = val_sec;
		nxQ = val_sec;
	}
	if (section == "ny") {
		j0 = val_sec;
		nyQ = val_sec;
	}
	if (section == "nz") {
		k0 = val_sec;
		nzQ = val_sec;
	}




	ofstream file((folder + "test.txt").c_str());


	int l;

	file << "x, y, i, j, L, vx, vy, ux, uy, phi, p, rho, H, eta" << endl;

	//all << setprecision(16) << fixed;

	for (int i = i0; i <= nxQ; i++) {
		for (int j = j0; j <= nyQ; j++) {
			for (int k = k0; k <= nzQ; k++) {
				l = i + off*j + off2*k;
				file << i*hx << " " << j*hy << " " << i << " " << j << " " << k << " " << l << " " << vx_h[l] << " " << vy_h[l] << " " << vz_h[l] 
					<< " " << C_h[l] << " " << T_h[l] << " " << p_h[l] 
					<< endl;
			}
		}
	}
}

__global__ void testCUDA() {

	printf("\n thread x:%i y:%i, information copied from device:\n", threadIdx.x, threadIdx.y);


	printf("\n");
}


int main(int argc, char **argv) {
	Log << "Compilation time: " << __DATE__ << " " << __TIME__ << endl;
	Log << "Run time start: " << get_time();
	Log << "command line: " << endl; for (int i = 0; i < argc; i++) Log << i << ": " << argv[i] << endl;
	//allocate heap size
	float heap_GB = 1.0;
	size_t limit = (size_t)(1024 * 1024 * 1024 * heap_GB);
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit);


	int devID = 0;
	cudaSetDevice(devID);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, devID);
	printf("\nDevice %d: \"%s\"\n", devID, deviceProp.name);
	Log << "GPU: " <<  devID << " " << deviceProp.name << endl;

	setParameters(argc, argv); 
	grid_near_border(1);
	
	//bettaC = 0;
	auxiliarySetup();
	 

#ifdef level2
	dim3 gridD(ceil((nx + 1.0) / thread_x_d), ceil((ny + 1.0) / thread_y_d));
	dim3 blockD(thread_x_d, thread_y_d);
#endif 

#ifdef level3
	dim3 gridD(ceil((nx + 1.0) / thread_x_d), ceil((ny + 1.0) / thread_y_d), ceil((nz + 1.0) / thread_z_d));
	dim3 blockD(thread_x_d, thread_y_d, thread_z_d);
#endif


	Poisson_setup();
	allocationCPU();
	
	read_recovery = 0;
	reading_recover(read_recovery);
	allocationGPU();

	//initial_level(true);


//syda_go: // if we reach a state that is considered to be stationary we go to this go-to-mark
	check
	write_fields(1, 1, 1, "xy_mid", nz / 2, "nz"); 
	write_fields(1, 1, 1, "xz_mid", ny / 2, "ny");
	write_fields(1, 1, 1, "yz_mid", nx / 2, "nx");

start:
next:
	
	copyConstantsGPU();


	

	hello << <1, 1 >> > ();
	cudaDeviceSynchronize();

	

	timer1 = clock() / CLOCKS_PER_SEC;

	testCUDA << <1, 1 >> > ();
	cudaDeviceSynchronize();
	cudaCheckError();
	//return 0;



	//pause
	// the main time loop of the whole calculation procedure
	while (true) {
		copied = 0;
		//if (timeq > 5000) return 0;
		iter = iter + 1; 	timeq = timeq + tau;
		
		//1st step, calculating of time evolutionary parts of velocity (quasi-velocity) , temperature and concentration 
		{

			//quasi_velocity << < gridD, blockD >> > (ux_d, uy_d, uz_d, vx_d, vy_d, vz_d, T_d, T0_d, C_d, C0_d);
			HeatEquation << < gridD, blockD >> > (vx_d, vy_d, vz_d, T_d, T0_d);
			//Concentration << < gridD, blockD >> > (vx_d, vy_d, vz_d, T_d, T0_d, C_d, C0_d);
			//Concentration_bc << < gridD, blockD >> > (T0_d, C_d);
			swap_one << < ceil(size_l / 1024.0), 1024 >> > (T0_d, T_d);
			//swap_one << < ceil(size_l / 1024.0), 1024 >> > (C0_d, C_d);
		}

		//pressure_point << <1,1 >> > (p0_d);

		//pressure_point << <1, 1 >> > (p0_d, nx / 2, ny / 2, 0.0);
		//2nd step, Poisson equation for pressure 
		if (iter % 1 == 0)
		{
			eps = 1.0; 		psiav0 = 0.0;		psiav = 0.0;		k = 0;
			while (eps > eps0*psiav0 || k < 2)
			{
				//pressure_norm1 << <1, 1 >> > (p0_d, nx, ny/2);
				//pressure_norm2 << < gridD, blockD >> > (p0_d);
				//pressure_point << <1, 1 >> > (p0_d, nx / 2, ny / 2, 1.0);
				psiav = 0.0;	k++;
				//Poisson << < gridD, blockD >> > (ux_d, uy_d, uz_d, p_d, p0_d );
				Poisson_Darcy << < gridD, blockD >> > (p_d, p0_d, T0_d, C0_d);
				//bc << < gridD, blockD >> > (p_d, T0_d, C0_d);
				//Poisson_Stream << < gridD, blockD >> > (p_d, p0_d, T0_d, C0_d);
				//if (timeq > 10) {	swap_one << < ceil(size_l / 1024.0), 1024 >> > (p0_d, p_d); break;}
				for (int i = 0; i < N_reduce; i++)
					reduction0 << < Grid_p[i], 1024, 1024 * sizeof(double) >> > (arr[i], N_p[i], arr[i + 1]);

				swap_one << < ceil(size_l / 1024.0), 1024 >> > (p0_d, p_d);
				cudaMemcpy(&psiav, psiav_array, sizeof(double), cudaMemcpyDeviceToHost);
				eps = abs(psiav - psiav0);		psiav0 = psiav;

				if (k % 1000 == 0) {
					cudaMemcpy(p_h, p_d, size_b, cudaMemcpyDeviceToHost);
					cout << k << "  " << setprecision(15) << p_h[1 + off] - p_h[nx - 1 + off*(ny - 1)] << " " << eps << endl;
				}

			}

		}
		Velocity_Darcy << < gridD, blockD >> > (vx_d, vy_d, vz_d, p0_d, T0_d, C0_d);
		//Velocity_Stream << < gridD, blockD >> > (vx_d, vy_d, vz_d, p0_d, T0_d, C0_d);
		//3
		//velocity_correction << < gridD, blockD >> > (ux_d, uy_d, uz_d, vx_d, vy_d, vz_d, p_d);
		//swap_5 << < ceil(size_l / 1024.0), 1024 >> >(ux_d, vx_d, uy_d, vy_d, uz_d, vz_d, C0_d, C_d, T0_d, T_d);
		//TEST		





		//4 




		//cout << "Hello" << endl;
		//TEST

			//4
#ifdef level3

		if (iter % (int)(tt * 1.0) == 0 || iter == 1 /*|| (iter % 100 == 0)*/) {
			//write_test();

			cout << setprecision(15) << endl;
			cout << fixed << endl;
			copyFromGPU();
			Ek_old = Ek;
			velocity();

			//cout << iter*tau << endl;
			cout << endl;
			cout << "Vx=" << maxval(vx_h, size_l) << endl;
			cout << "Vy=" << maxval(vy_h, size_l) << endl;
			cout << "Vz=" << maxval(vz_h, size_l) << endl;
			cout << "Vy_c10=" << vy_h[10 + off * 10 + off2 * 10] << endl;
			cout << "p=" << maxval(p_h, size_l) << endl;
			pmax = maxval(p_h, size_l);
			pmin = minval(p_h, size_l);
			//angular_momentum(vx_h, vy_h, vz_h, nx, ny, nz, hx, hy, hz, AMx, AMy, AMz, AMabs);
			cout << "p_iter=" << k << endl;
			cout << "Vmax= " << Vmax << endl;
			cout << setprecision(20);

			cout << "Ek= " << Ek << endl;
			cout << "dEk= " << (Ek - Ek_old) << endl;
			cout << "dEk/Ek= " << abs(Ek - Ek_old) / Ek << endl;
			dCmax0 = dCmax;  dCmax = max_vertical_difference(C_h);
			dTmax0 = dTmax;  dTmax = max_vertical_difference(T_h);
			cout << "max_dT= " << dTmax << endl;
			cout << "max_dC= " << dCmax << endl;
			cout << setprecision(7) << "t=" << timeq << endl;
			timer


				if (iter == 1)
				{
					integrals << "t, Ek, Vmax,  time(min), Vc,  dEk, dTmax, dCmax"
						<< ", NuFront, NuBack"
						<< ", NuDown, NuTop"
						<< ", ShFront, ShBack"
						<< ", ShDown, ShTop"
						<< " " << Ra*deltaTemp
						<< endl;
				}

			integrals << setprecision(20) << fixed;
			integrals << tau*iter << " " << Ek << " " << Vmax << " " << (timer2 - timer1) / 60 << " " << vy_h[nx / 2 + off * ny / 2 + off2 * nz / 2] << " " << (Ek - Ek_old) << " "
				//<< Nu_y_top(T_h) << " " << Nu_y_down(T_h) << " " << Nu_x_left(T_h) << " " << Nu_x_right(T_h) 
				<< dTmax << " " << dCmax
				<< " " << Nu_z_front(T_h) << " " << Nu_z_back(T_h)
				<< " " << Nu_y_down(T_h) << " " << Nu_y_top(T_h)

				<< " " << Nu_z_front(C_h) << " " << Nu_z_back(C_h)
				<< " " << Nu_y_down(C_h) << " " << Nu_y_top(C_h)
				<< endl;

			//printf("%30.25f \n", Ek); pause


			if (1)
				if (timeq > minimumTime && abs(Ek - Ek_old) / Ek < 1e-7 && (dCmax - dCmax0) / dCmax < 1e-8) {
					Ra_tab << Ra*deltaTemp << " " << Ek << " " << Vmax << " " << dCmax << " " << Ctotal << " " << (timer2 - timer1) / 60 << " " << timeq
						<< endl;
					next_to = 1;
				}


			if (Ek != Ek) exit(0);
			if (Ra*deltaTemp > 1000) exit(0);
		}

		if (iter % (int)(tt * 0.5) == 0 || iter == 1 || next_to == 1) {
			if (dimensionless) {
				transform(C_h, C_h, coefC, shiftC);
				transform(T_h, T_h, coefT, shiftT);
			}
			write_fields(1, 1, 1, "all");
			write_fields(1, 1, 1, "xy_mid", nz / 2, "nz");
			write_fields(1, 1, 1, "xz_mid", ny / 2, "ny");
			write_fields(1, 1, 1, "yz_mid", nx / 2, "nx");
			//write_fields_para(1, 1, "xy_mid", nz / 2, "nz");
			if (next_to == 1) {
				next_to = 0;
				iter = 0;
				timeq = 0;
				if (dimensionless) {
					Ra += dT;
				}
				else {
					T_hot += dT;
					deltaTemp = T_hot - T_cold;
				}
				goto next;
			}
		}
		if (next_to == 1) {
				write_fields(1, 1, 1, "F_all", nz / 2, "nz");
				write_fields(1, 1, 1, "F_section", nx / 2, "nx");
				next_to = 0;
				iter = 0;
				timeq = 0;
				if (dimensionless) {
					Ra += dT;
				}
				else {
					T_hot += dT;
					deltaTemp = T_hot - T_cold;
				}
				if (Ra*deltaTemp >= StopIncr) return 0;
				goto next;
		}


#endif // level3

#ifdef level2
		if (iter % (int)(tt * 1.0) == 0 || iter == 1 /*|| (iter % 100 == 0)*/) {
			//write_test();
			copyFromGPU();
			cout << setprecision(15) << endl;
			cout << fixed << endl;
			
			Ek_old = Ek;
			velocity();

			//cout << iter*tau << endl;
			cout << endl;
			cout << "Vx=" << maxval(vx_h, size_l) << endl;
			cout << "Vy=" << maxval(vy_h, size_l) << endl;
			cout << "p_iter=" << k << endl;
			cout << "Vmax= " << Vmax << endl;
			cout << setprecision(20);

			cout << "Ek= " << Ek << endl;
			cout << "dEk= " << (Ek - Ek_old) << endl;
			cout << "dEk/Ek= " << abs(Ek - Ek_old) / Ek << endl;
			dCmax0 = dCmax;  dCmax = max_vertical_difference(C_h);
			dTmax0 = dTmax;  dTmax = max_vertical_difference(T_h);
			Cmax = maxval(C_h, size_l); Cmin = minval(C_h, size_l);
			pmax = maxval(p_h, size_l); pmin = minval(p_h, size_l);
			Ctotal = Integral(C_h); Ttotal = Integral(T_h); Ptotal = Integral(p_h);
			cout << "max_dT= " << dTmax << endl;
			cout << "max_dC= " << dCmax << endl;
			cout << "Cmax= " << Cmax << endl;
			cout << "Cmin= " << Cmin << endl;
			cout << "Ctotal= " << Ctotal << endl;
			cout << setprecision(7) << "t=" << timeq << endl;
			timer


				if (iter == 1)
				{
					integrals << "t, Ek, Vmax,  time(min), Vc,  dEk, dTmax, dCmax, Ctotal, Ttotal, Cmax, Cmin, pmax, pmin, dP, Ptotal"
						<< " " << Ra*deltaTemp
						<< endl;
				}

			integrals << setprecision(20) << fixed;
			integrals << tau*iter << " " << Ek << " " << Vmax << " " << (timer2 - timer1) / 60 << " " << vy_h[nx / 2 + off * ny / 2 + off2 * nz / 2] << " " << (Ek - Ek_old) << " "
				//<< Nu_y_top(T_h) << " " << Nu_y_down(T_h) << " " << Nu_x_left(T_h) << " " << Nu_x_right(T_h) 
				<< dTmax << " " << dCmax << " " << Ctotal << " " << Ttotal << " " << Cmax << " " << Cmin << " " << pmax << " " << pmin << " " << pmax - pmin << " " << Ptotal
				<< endl;

			//printf("%30.25f \n", Ek); pause


			if (1)
				if (timeq > minimumTime && abs(Ek - Ek_old) / Ek < 1e-7 && (dCmax - dCmax0) / dCmax < 1e-8) {
					Ra_tab << Ra*deltaTemp << " " << Ek << " " << Vmax << " " << dCmax << " " << Ctotal << " " << (timer2 - timer1) / 60 << " " << timeq
						<< endl; 
					next_to = 1;
				}


			if (Ek != Ek) exit(0);
			//if (Ra*deltaTemp > 1000) exit(0);
		}

		if (iter % (int)(tt * 5) == 0 || iter == 1 || next_to == 1) {
			if (dimensionless) {
				transform(C_h, C_h, coefC, shiftC);
				transform(T_h, T_h, coefT, shiftT);
			}
			write_fields(1, 1, 1, "all", nz / 2, "nz");
			write_fields(1, 1, 1, "section", nx / 2, "nx");
			//write_fields_para(1, 1, "xy_mid", nz / 2, "nz");
			if (next_to == 1) {
				write_fields(1, 1, 1, "F_all", nz / 2, "nz");
				write_fields(1, 1, 1, "F_section", nx / 2, "nx");
				next_to = 0;
				iter = 0;
				timeq = 0;
				if (dimensionless) {
					Ra += dT;
				}
				else {
					T_hot += dT;
					deltaTemp = T_hot - T_cold;
				}
				if (Ra*deltaTemp >= StopIncr) return 0;
				goto next;
			}
		}
#endif // level2



		// the end of 4





	} //the end of the main loop



	return 0;
}










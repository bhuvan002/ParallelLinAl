#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include "mat_utils.h"
#include "matrix.h"


using namespace std;

template <typename T>
struct absolute {

	__device__
	T operator()(T &x) const {
		if (x < 0) {
			return -x;
		} else {
			return x;
		}
	}
};

__global__
void find_ratio_and_zero(float *A, int N, int M, int i, float *ratio) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= (N-(i+1))) return;

	int k = tid + (i+1);
	ratio[k] = A[idx(k,i,M)]/A[idx(i,i,M)];
	A[idx(k,i,M)] = 0;
}

__global__
void reduce_submatrix(float *A, int N, int M, int i, float *ratio) {
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	int jj = blockIdx.y * blockDim.y + threadIdx.y;
	if (ii >= (N-(i+1)) || jj >= (M-(i+1))) return;

	int k = ii + (i+1);
	int j = jj + (i+1);
	A[idx(k,j,M)] -= ratio[k] * A[idx(i,j,M)];
}

__global__
void reduce_rhs(float *A, int N, int M, int i, float *X) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid > i) return;

	int k = tid;
	A[idx(k,N,M)] -= A[idx(k,i,M)] * X[i];	
}

/*
	h_A -> 1-d array of NxM matrix
	X -> 1-d array, host alloc'ed
	solve -> 1 if h_x is not null
*/
void ge_parallel(float *h_A, float *h_row_echelon_A, int N, int M, float *X, int reduced) {
	float *d_A;
	float *d_vec; // N-dim
	float *d_ratio;

	cudaMalloc(&d_A, N*M*sizeof(float));
	cudaMalloc(&d_vec, N*sizeof(float));
	cudaMalloc(&d_ratio, N*sizeof(float));

	cudaMemcpy(d_A, h_A, N*M*sizeof(float), cudaMemcpyHostToDevice);

	dim3 threads(16, 16);
	dim3 blocks((N+15)/16, (M+15)/16);

	cudaDeviceSynchronize();

	if (0) {
		float h_vec[N];
		cudaMemcpy(&h_vec[0], d_vec, N*sizeof(float), cudaMemcpyDeviceToHost);

		for (int j=0; j<N; j++) {
			printf("%f ", h_vec[j]);
		}
		printf("\n");
	}

	thrust::device_ptr<float> th_A = thrust::device_pointer_cast(d_A);
	thrust::device_ptr<float> th_vec = thrust::device_pointer_cast(d_vec);

	thrust::device_ptr<float> th_x = thrust::device_pointer_cast(X);

	thrust::device_vector<float> tmp(M);


	int max_row;
	for (int i=0; i<N; i++) {
		// Search for maximum in the ith column
		copy_abs_col_to_vec<<<num_blocks(N,512),512>>>(d_A, d_vec, N, M, i);
		cudaDeviceSynchronize();

		// thrust::copy(th_vec+idx(i,i,N), th_vec+idx(i,0,N)+N, std::ostream_iterator<int>(std::cout, "\n"));

		thrust::device_vector<float>::iterator iter = 
			thrust::max_element(thrust::device, th_vec+i, th_vec+N);
		cudaDeviceSynchronize();

		max_row = thrust::device_pointer_cast(&(iter[0]))-(th_vec);

		// Swap ith row with the maximum row (for numerical stability)
		thrust::copy(thrust::device, th_A+idx(i,0,M), th_A+idx(i+1,0,M), tmp.begin());
		cudaDeviceSynchronize();
		thrust::copy(thrust::device, th_A+idx(max_row,0,M), th_A+idx(max_row+1,0,M), th_A+idx(i,0,M));
		cudaDeviceSynchronize();
		thrust::copy(thrust::device, tmp.begin(), tmp.end(), th_A+idx(max_row,0,M));
		cudaDeviceSynchronize();

		// P;

		// Zero all entries in the ith column below the ith row
		find_ratio_and_zero<<<num_blocks(N-(i+1), 512), 512>>>(d_A, N, M, i, d_ratio);
		cudaDeviceSynchronize();

		// P;

		dim3 threads(16, 16);
		dim3 blocks(num_blocks(N-(i+1),16), num_blocks(M-(i+1),16));
		reduce_submatrix<<<threads, blocks>>>(d_A, N, M, i, d_ratio);

		// P;

		cudaDeviceSynchronize();
	}

	// P;

	if (h_row_echelon_A != NULL) {
		cudaMemcpy(h_row_echelon_A, d_A, N*M*sizeof(float), cudaMemcpyDeviceToHost);
	}

	// P;

	if (X != NULL) {
		thrust::fill(thrust::device, th_x, th_x+N, 0);
		cudaDeviceSynchronize();

		for (int i=N-1; i>=0; i--) {
			X[i] = th_A[idx(i,N,M)]/th_A[idx(i,i,M)];
			reduce_rhs<<<num_blocks(i,512), 512>>>(d_A, N, M, i, X);
			cudaDeviceSynchronize();
		}
	}

	if (reduced) {

		float X[N][M-N];
		for (int i=N-1; i>=0; i--) {
			for (int j=0; j<M-N; j++) {
				h_row_echelon_A[idx(i,N+j,M)] /= h_row_echelon_A[idx(i,i,M)];
				X[i][j] = h_row_echelon_A[idx(i,N+j,M)];

			}
			h_row_echelon_A[idx(i,i,M)] = 1;
			for (int k=i-1; k>=0; k--) {
				for (int j=0; j<M-N; j++) {
					h_row_echelon_A[idx(k,N+j,M)] -= h_row_echelon_A[idx(k,i,M)] * X[i][j];
				}
				h_row_echelon_A[idx(k,i,M)] = 0;
			}
		}
	}

	cudaFree(d_A);
	cudaFree(d_vec);
	cudaFree(d_ratio);
}
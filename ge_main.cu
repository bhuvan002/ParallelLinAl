#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include "mat_utils.h"
#include "matrix.h"

#define P printf("line: %d\n", __LINE__);


/*
	a -> 1-d array of a N by M matrix
	modifies "a" in place
	transforms the matrix into it's row echelon form
*/
void ge_cpu(float *a, float *re_a, int N, int M, float *x) {
	float max_elem;
	int max_row;

	memcpy(re_a, a, N*M*sizeof(float));

	for (int i=0; i<N; i++) {
		// Search for maximum
		max_elem = abs(re_a[idx(i,i,M)]);
		max_row = i;
		for (int k=i+1; k<N; k++) {
			float elem = abs(re_a[idx(k,i,M)]);
			if (elem > max_elem) {
				max_elem = elem;
				max_row = k;
			}
		}

		// Swap ith row with the maximum row (for numerical stability)
		float tmp;
		for (int k=i; k<M; k++) {
			tmp = re_a[idx(max_row,k,M)];
			re_a[idx(max_row,k,M)] = re_a[idx(i,k,M)];
			re_a[idx(i,k,M)] = tmp;
		}

		// Zero all entries in the ith column below the ith row
		for (int k=i+1; k<N; k++) {
			float ratio = re_a[idx(k,i,M)]/re_a[idx(i,i,M)];
			re_a[idx(k,i,M)] = 0;
			for (int j=i+1; j<M; j++) {
				re_a[idx(k,j,M)] -= ratio * re_a[idx(i,j,M)];
			}
		}
	}

	if (x != NULL) {
		for (int i=0; i<N; i++) {
			x[i] = 0;
		}

		for (int i=N-1; i>=0; i--) {
			x[i] = re_a[idx(i,N,M)]/re_a[idx(i,i,M)];
			for (int k=i-1; k>=0; k--) {
				re_a[idx(k,N,M)] -= re_a[idx(k,i,M)] * x[i];
			}
		}
	}
}

template <typename T>
struct absolute {

	__device__
	void operator()(T &x) const {
		if (x < 0) {
			x = -x;
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
void ge_parallel(float *h_A, float *h_row_echelon_A, int N, int M, float *X) {
	float *d_A;
	float *d_A_t; // M*N matrix
	float *d_ratio;

	cudaMalloc(&d_A, N*M*sizeof(float));
	cudaMalloc(&d_A_t, M*N*sizeof(float));
	cudaMalloc(&d_ratio, N*sizeof(float));

	cudaMemcpy(d_A, h_A, N*M*sizeof(float), cudaMemcpyHostToDevice);

	dim3 threads(16, 16);
	dim3 blocks((N+15)/16, (M+15)/16);
	matrix_transpose_gpu<<<blocks, threads>>>(d_A, d_A_t, N, M);

	cudaDeviceSynchronize();

	if (0) {
		float h_A_t[M][N];
		cudaMemcpy(&h_A_t[0][0], d_A_t, M*N*sizeof(float), cudaMemcpyDeviceToHost);

		for (int i=0; i<M; i++) {
			for (int j=0; j<N; j++) {
				printf("%f ", h_A_t[i][j]);
			}
			printf("\n");
		}
	}

	thrust::device_ptr<float> th_A = thrust::device_pointer_cast(d_A);
	thrust::device_ptr<float> th_A_t = thrust::device_pointer_cast(d_A_t);

	thrust::device_ptr<float> th_x = thrust::device_pointer_cast(X);

	thrust::device_vector<float> tmp(M);


	int max_row;
	for (int i=0; i<N; i++) {
		// Search for maximum in the ith column

		thrust::for_each(thrust::device, th_A_t+idx(i,i,N), th_A_t+idx(i+1,0,N), absolute<float>());
		cudaDeviceSynchronize();

		thrust::copy(th_A_t+idx(i,i,N), th_A_t+idx(i,0,N)+N, std::ostream_iterator<int>(std::cout, "\n"));

		thrust::device_vector<float>::iterator iter = 
			thrust::max_element(thrust::device, th_A_t+idx(i,i,N), th_A_t+idx(i+1,0,N));
		cudaDeviceSynchronize();

		max_row = thrust::device_pointer_cast(&(iter[0]))-(th_A_t+idx(i,0,N));


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

	cudaFree(d_A);
	cudaFree(d_A_t);
	cudaFree(d_ratio);
}

int main() {
	int N, M;

	scanf("%d %d", &N, &M);
	float A[N][M];
	float row_echelon_A[N][M];
	float *X;

	cudaHostAlloc(&X, N*sizeof(float), 0);

	for (int i=0; i<N; i++) {
		for (int j=0; j<M; j++) {
			scanf("%f", &A[i][j]);
		}
	}

	if (1) {
		ge_cpu((float *) A, (float *) row_echelon_A, N, M, NULL);
	}

	if (1) {
		for (int i=0; i<N; i++) {
			for (int j=0; j<M; j++) {
				printf("%f ", row_echelon_A[i][j]);
			}
			printf("\n");
		}
	}

	if (1) {
		ge_parallel((float *) A, (float *) row_echelon_A, N, M, X);
		cudaDeviceSynchronize();
	}

	if (1) {
		for (int i=0; i<N; i++) {
			for (int j=0; j<M; j++) {
				printf("%f ", row_echelon_A[i][j]);
			}
			printf("\n");
		}
	}

	if (1) {
		for (int i=0; i<N; i++) {
			printf("%f ", X[i]);
		}
		printf("\n");
	}
	return 0;
}
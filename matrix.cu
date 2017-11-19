#include <stdlib.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include "matrix.h"
#include "mat_utils.h"

void matrix_multiply( float *h_A,  float *h_B, float *h_C, int N, int K, int M) {
	float *d_A, *d_B, *d_B_t, *d_C;
	cudaMalloc(&d_A, N*K*sizeof(float));
	cudaMalloc(&d_B, K*M*sizeof(float));
	cudaMalloc(&d_B_t, K*M*sizeof(float));
	cudaMalloc(&d_C, N*M*sizeof(float));

	cudaMemcpy(d_A, h_A, N*K*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, K*M*sizeof(float), cudaMemcpyHostToDevice);

	dim3 threads(16, 16);
	dim3 blocks1((K+15)/16, (M+15)/16);
	matrix_transpose_gpu<<<blocks1, threads>>>(d_B, d_B_t, K, M);

	dim3 blocks2((N+15)/16, (M+15)/16);
	matrix_multiply_gpu_fast<<<blocks2, threads>>>(d_A, d_B_t, d_C, N, K, M);

	cudaMemcpy(h_C, d_C, N*M*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_B_t);
	cudaFree(d_C);
}

/* 
	Multiplies matrices C = A x B
	A   : N x K
	B_t : M x K (transpose of B)
	C   : N x M
*/
__global__ void matrix_multiply_gpu_fast(float *A, float *B_t, float *C, int N, int K, int M) {
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	int jj = blockIdx.y * blockDim.y + threadIdx.y;
	if (ii >= N || jj >= M) return;

	thrust::device_ptr<float> d_A = thrust::device_pointer_cast(A);
	thrust::device_ptr<float> d_B = thrust::device_pointer_cast(B_t);

	C[ii*M + jj] = thrust::inner_product(thrust::device, d_A+ii*K, d_A+ii*K+K, d_B+jj*K, 0.0f);
}

/*  
	Multiplies two matrices C = A x B
	A should have dimensions N x K
	B should have dimensions K x M
	C will have dimensions N x M
*/
__global__ void matrix_multiply_gpu(float *A, float *B, float *C, int N, int K, int M) {
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	int jj = blockIdx.y * blockDim.y + threadIdx.y;
	if (ii >= N || jj >= M) return;
	C[ii*M + jj] = 0.0f;
	for (int kk = 0; kk < K; kk++) {
		C[ii*M + jj] += A[ii*K + kk] * B[kk*M + jj];
	}
}

/*
	Computes the transpose of a matrix A
	of dimensions N x M and stores
	the result in A_t (M x N)
*/
__global__ void matrix_transpose_gpu(float *A, float *A_t, int N, int M) {
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	int jj = blockIdx.y * blockDim.y + threadIdx.y;
	if (ii >= N || jj >= M) return;
	A_t[jj*N + ii] =A[ii*M + jj];
}


__global__ void matrix_sub_gpu(float *A, float *B, float *C, int M, int N) {
	int ii = blockIdx.x * blockDim.x + threadIdx.x;
	int jj = blockIdx.y * blockDim.y + threadIdx.y;
	if (ii >= M || jj >= N) return;

	int pos = idx(ii,jj,N);
	C[pos] = A[pos] - B[pos];
}

/*
	Stores A-B (MxN) in C	
*/
void matrix_sub(float *h_A, float *h_B, float *h_C, int M, int N) {
	float *d_A, *d_B, *d_C;

	cudaMalloc(&d_A, M*N*sizeof(float));
	cudaMalloc(&d_B, M*N*sizeof(float));
	cudaMalloc(&d_C, M*N*sizeof(float));

	cudaMemcpy(d_A, h_A, M*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, M*N*sizeof(float), cudaMemcpyHostToDevice);

	dim3 threads(16, 16);
	dim3 blocks((M+15)/16, (N+15)/16);
	matrix_transpose_gpu<<<blocks, threads>>>(d_A, d_B, d_C, M, N);
	cudaDeviceSynchronize();

	cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
#include "givens.h"
#include "matrix.h"
#include <math.h>
#include <stdio.h>

__host__ __device__
void givens(float a, float b, float *c, float *s, float *r) {
	float h, d;
	h = hypotf(a, b);
	d = 1.0f/h;
	*c = fabsf(a)*d;
	*s = copysignf(d, a)*b;
	*r = copysignf(1.0f, a)*h;
}

__global__ void print_matrix(float *A, int M, int N) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			printf("%f ", A[i*N + j]);
		}
		printf("\n");
	}
	printf("*-------------------*\n\n");
}

__global__ void print_matrix_transpose(float *A, int M, int N) {
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < M; i++) {
			printf("%f ", A[i*N + j]);
		}
		printf("\n");
	}
	printf("*-------------------*\n\n");
}


/*
	Performs QR factorization of A [M x N] and stores
	the results in Q [M x M] and R [M x N]
*/
void givens_rotation(float *A, float *Q, float *R, int M, int N) {
	float *d_A, *d_Q, *d_R_t, *d_A_t;
	cudaMalloc(&d_Q, M*M*sizeof(float));
	cudaMalloc(&d_R_t, M*N*sizeof(float));
	cudaMalloc(&d_A, M*N*sizeof(float));
	cudaMalloc(&d_A_t, M*N*sizeof(float));

	cudaMemcpy(d_A, A, M*N*sizeof(float), cudaMemcpyHostToDevice);

	// print_matrix<<<1,1>>>(d_A, M, N);
	// cudaDeviceSynchronize();

	dim3 threads(16, 16);
	dim3 blocks1((M+15)/16, (M+15)/16);
	dim3 blocks2((M+15)/16, (N+15)/16);
	dim3 blocks3((N+15)/16, (M+15)/16);

	matrix_transpose_gpu<<<blocks2, threads>>>(d_A, d_A_t, M, N);

	identity<<<blocks1, threads>>>(d_Q, M);
	cudaMemcpy(d_R_t, d_A_t, M*N*sizeof(float), cudaMemcpyDeviceToDevice);

	// TODO
	// print_matrix<<<1,1>>>(d_Q, M, M);
	// cudaDeviceSynchronize();

	for (int j = 0; j < N; j++) {
		for (int i = M-1; i >= j+1; i--) {
			float a, b;
			cudaMemcpy(&a, d_R_t+j*M+(i-1), sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(&b, d_R_t+j*M+i, sizeof(float), cudaMemcpyDeviceToHost);

			// TODO
			//printf("a = %f, b = %f\n", a, b);

			float c, s, r;
			givens(a, b, &c, &s, &r);
			//TODO
			// printf("a = %f, b = %f, c = %f, s = %f, r = %f\n", a, b, c, s, r);

			givens_rotate_R<<<(N+15)/16, 16>>>(d_R_t, M, N, i, j, c, s, r);
			givens_rotate_Q<<<(M+15)/16, 16>>>(d_Q, M, N, i, c, s);

			
			// print_matrix_transpose<<<1,1>>>(d_R_t, N, M);
			// cudaDeviceSynchronize();
		}
	}
	cudaMemcpy(d_A_t, d_R_t, M*N*sizeof(float), cudaMemcpyDeviceToDevice);
	matrix_transpose_gpu<<<blocks3, threads>>>(d_A_t, d_R_t, N, M);

	cudaMemcpy(R, d_R_t, M*N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(Q, d_Q, M*M*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_Q);
	cudaFree(d_R_t);
	cudaFree(d_A);
	cudaFree(d_A_t);

}

__global__ void givens_rotate_R(float *R_t, int M, int N, int i, int rot_col, 
		float c, float s, float r) {

	int j = blockIdx.x*blockDim.x + threadIdx.x;
	if (j < rot_col || j >= N) return;
	if (j == rot_col) {
		R_t[j*M + i - 1] = r;
		R_t[j*M + i] = 0.0f;
	}
	else {
		float a = R_t[j*M + i - 1];
		float b = R_t[j*M + i];
		R_t[j*M + i - 1] = c*a + s*b;
		R_t[j*M + i] = -s*a + c*b;
		// printf("j = %d, a = %f and b = %f\n", j, R_t[j*M + i - 1], R_t[j*M + i]);
	}
}

__global__ void givens_rotate_Q(float *Q, int M, int N, int j, float c, float s) {

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= M) return;
	float a = Q[i*M + j-1];
	float b = Q[i*M + j];
	Q[i*M + j-1] = c*a + s*b;
	Q[i*M + j] = -s*a + c*b;
}
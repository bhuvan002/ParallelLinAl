#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include "conjgrad.h"
#include "matrix.h"
#include <math.h>
#include <stdio.h>

/*

	Initializes a vector to zeros
*/
__global__ void zeros(float *A, int N) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < N) {
		A[idx] = 0.0f;
	}
}

__global__ void dprint(float *A, int N) {
	for (int i = 0; i < N; i++) {
		printf("%f ", A[i]);
	}
	printf("\n");
}

__global__ void dprint2(float *A, int M, int N) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++)
			printf("%f ", A[i*N + j]);
		printf("\n");
	}
}


/*
	Computes solution to the system of
	linear equations Ax=b using conjugate
	gradient methods. The matrix A should be
	symmetric and positive-definite.
*/
void conjgrad(float *A, float *x, float *b, int N) {
	float *d_A, *d_x, *d_b, *d_p, *d_r, *d_temp, *d_mattemp;
	cudaMalloc(&d_A, N*N*sizeof(float));
	cudaMalloc(&d_mattemp, N*N*sizeof(float));
	cudaMalloc(&d_x, N*sizeof(float));
	cudaMalloc(&d_b, N*sizeof(float));
	cudaMalloc(&d_p, N*sizeof(float));
	cudaMalloc(&d_r, N*sizeof(float));
	cudaMalloc(&d_temp, N*sizeof(float));


	cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyHostToDevice);


	dim3 threads(16, 16);
	dim3 blocks((N+15)/16, (N+15)/16);
	matrix_transpose_gpu<<<blocks, threads>>>(d_A, d_mattemp, N, N);

	zeros<<<(N+15)/16, 16>>>(d_x, N);
	// dprint<<<1, 1>>>(d_x, N);
	// cudaDeviceSynchronize();

	x[0] = 2; x[1] = 1;
	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);

	// print<<<(N+15)/16, 16>>>(x, N);
	// cudaDeviceSynchronize();

	matrix_vector_multiply<<<(N+15)/16, 16>>>(d_A, d_x, d_temp, N);
	// dprint<<<1, 1>>>(d_x, N);
	// cudaDeviceSynchronize();
	// dprint<<<1, 1>>>(d_temp, N);
	// cudaDeviceSynchronize();
	
	thrust::device_ptr<float> pd_b = thrust::device_pointer_cast(d_b);
	thrust::device_ptr<float> pd_x = thrust::device_pointer_cast(d_x);
	thrust::device_ptr<float> pd_r = thrust::device_pointer_cast(d_r);
	thrust::device_ptr<float> pd_temp = thrust::device_pointer_cast(d_temp);
	thrust::device_ptr<float> pd_p = thrust::device_pointer_cast(d_p);

	thrust::transform(pd_temp, pd_temp+N, pd_b, pd_r, saxpy_functor(-1.0f));
	cudaMemcpy(d_p, d_r, N*sizeof(float), cudaMemcpyDeviceToDevice);
	float rr_old = thrust::inner_product(pd_r, pd_r + N, pd_r, 0.0f);

	int kk = 0;
	while (kk < 10*N) {
		matrix_vector_multiply<<<(N+15)/16, 16>>>(d_A, d_p, d_temp, N);
		float pp = thrust::inner_product(pd_temp, pd_temp+N, pd_p, 0.0f);
		float alpha = rr_old/pp;
		thrust::transform(pd_p, pd_p+N, pd_x, pd_x, saxpy_functor(alpha));
		matrix_vector_multiply<<<(N+15)/16, 16>>>(d_A, d_p, d_temp, N);
		thrust::transform(pd_temp, pd_temp+N, pd_r, pd_r, saxpy_functor(-alpha));
		float rr_new = thrust::inner_product(pd_r, pd_r + N, pd_r, 0.0f);
		if(sqrtf(rr_new) < 1E-6) {
			break;
		}
		thrust::transform(pd_p, pd_p+N, pd_r, pd_p, saxpy_functor(rr_new/rr_old));
		rr_old = rr_new;
		kk++;
	}

	cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(d_A);
	cudaFree(d_mattemp);
	cudaFree(d_x);
	cudaFree(d_b);
	cudaFree(d_p); 
	cudaFree(d_r); 
	cudaFree(d_temp);
}

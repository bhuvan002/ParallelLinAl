#include <stdio.h>
#include "mat_utils.h"

void copy_sub_matrix(float *A, float *sub_A, int M, int N, int r, int c) {
	// int sub_M = M-r;
	int sub_N = N-c;

	for (int i=r; i<M; i++) {
		for (int j=c; j<N; j++) {
			sub_A[idx(i-r,j-c,sub_N)] = A[idx(i,j,N)];
		}
	}
}

void copy_back_sub_matrix(float *A, float *sub_A, int M, int N, int r1, int c1, int r2, int c2) {
	// int sub_M = M-r;
	int sub_N = c2-c1+1;

	for (int i=r1; i<=r2; i++) {
		for (int j=c1; j<=c2; j++) {
			A[idx(i,j,N)] = sub_A[idx(i-r1,j-c1,sub_N)];
		}
	}
}

void scalar_mul(float *A, float *scaled_A, int scalar, int M, int N) {
	for (int i=0; i<M; i++) {
		for (int j=0; j<N; j++) {
			scaled_A[idx(i,j,N)] = A[idx(i,j,N)] * scalar;
		}
	}
}

void fill_val(float *A, float val, int M, int N) {
	for (int i=0; i<M; i++) {
		for (int j=0; j<N; j++) {
			A[idx(i,j,N)] = val;
		}
	}
}

void print_mat(float *A, int M, int N) {
	for (int i=0; i<M; i++) {
		for (int j=0; j<N; j++) {
			printf("%f ", A[idx(i,j,N)]);
		}
		printf("\n");
	}
}

void print_d_mat(float *d_A, int M, int N) {
	float h_A[M*N];
	cudaMemcpy(h_A, d_A, M*N*sizeof(float), cudaMemcpyDeviceToHost);

	print_mat(h_A, M, N);
}
#include <stdio.h>

__device__ __host__
inline int idx(int x, int y, int num_cols) {
	return x*num_cols+y;
}

__device__ __host__ 
inline int num_blocks(int tot_threads, int t_per_b) {
	return (tot_threads+t_per_b-1)/t_per_b;
}

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
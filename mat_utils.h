#ifndef MAT_UTILS_H
#define MAT_UTILS_H

__device__ __host__
inline int idx(int x, int y, int num_cols) {
	return x*num_cols+y;
}

__device__ __host__ 
inline int num_blocks(int tot_threads, int t_per_b) {
	return (tot_threads+t_per_b-1)/t_per_b;
}

void copy_sub_matrix(float *A, float *sub_A, int M, int N, int r, int c);

void copy_back_sub_matrix(float *A, float *sub_A, int M, int N, int r1, int c1, int r2, int c2);

void scalar_mul(float *A, float *scaled_A, int scalar, int M, int N);

void fill_val(float *A, float val, int M, int N);

void print_mat(float *A, int M, int N);
#endif
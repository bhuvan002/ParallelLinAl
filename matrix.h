#ifndef MATRIX_H
#define MATRIX_H

void matrix_multiply(float *h_A,  float *h_B, float *h_C, int N, int K, int M);
void matrix_sub(float *h_A, float *h_B, float *h_C, int M, int N);
__global__ void matrix_multiply_gpu_fast(float *A,  float *B_t, float *C, int N, int K, int M);
__global__ void matrix_multiply_gpu(float *A,  float *B, float *C, int N, int K, int M);
__global__ void matrix_transpose_gpu(float *A, float *A_t, int N, int M);
__global__ void matrix_vector_multiply(float *A, float*b, float*c, int N);
__global__ void matrix_sub_gpu(float *A, float *B, float *C, int M, int N);
__global__ void identity(float *A, int N);
#endif

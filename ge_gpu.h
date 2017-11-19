#ifndef GE_GPU_H
#define GE_GPU_H

void ge_parallel(float *h_A, float *h_row_echelon_A, int N, int M, float *X, int reduced);

#endif
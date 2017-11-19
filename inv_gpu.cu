#include <stdio.h>
#include <iostream>
#include "ge_gpu.h"
#include "mat_utils.h"

using namespace std;

void inverse(float *A, float *inv_A, int M) {
	float aug_A[M][2*M];
	fill_val((float *) aug_A, 0, M, 2*M);

	copy_back_sub_matrix((float *) aug_A, (float *) A, M, 2*M, 0, 0, M-1, M-1);
	for (int i=0; i<M; i++) {
		aug_A[i][M+i] = 1;
	}

	float re_aug_A[M][2*M];
	ge_parallel((float *) aug_A, (float *) re_aug_A, M, 2*M, NULL, 1);
	copy_sub_matrix((float *) re_aug_A, (float *) inv_A, M, 2*M, 0, M);
}
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
#include "ge_gpu.cu"

#define P printf("line: %d\n", __LINE__);


/*
	a -> 1-d array of a N by M matrix
	modifies "a" in place
	transforms the matrix into it's row echelon form
*/
void ge_cpu(float *a, float *re_a, int N, int M, float *x, int reduced) {
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

	if (reduced) {
		float X[N][M-N];
		for (int i=N-1; i>=0; i--) {
			for (int j=0; j<M-N; j++) {
				re_a[idx(i,N+j,M)] /= re_a[idx(i,i,M)];
				X[i][j] = re_a[idx(i,N+j,M)];

			}
			re_a[idx(i,i,M)] = 1;
			for (int k=i-1; k>=0; k--) {
				for (int j=0; j<M-N; j++) {
					re_a[idx(k,N+j,M)] -= re_a[idx(k,i,M)] * X[i][j];
				}
				re_a[idx(k,i,M)] = 0;
			}
		}
	}
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

	if (0) {
		ge_cpu((float *) A, (float *) row_echelon_A, N, M, NULL, 1);
	}

	if (0) {
		for (int i=0; i<N; i++) {
			for (int j=0; j<M; j++) {
				printf("%f ", row_echelon_A[i][j]);
			}
			printf("\n");
		}
	}

	if (1) {
		ge_parallel((float *) A, (float *) row_echelon_A, N, M, NULL, 1);
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
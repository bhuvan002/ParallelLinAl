#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "givens.h"

int main() {
	int M, N;
	// printf("Enter N, K, M: ");
	scanf("%d %d", &M, &N);
	float A[M][N];

	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			scanf("%f", &A[i][j]);
		}
	}

	float Q[M][M], R[M][N];

	givens_rotation((float *)A, (float *)Q, (float *)R, M, N);
	cudaDeviceSynchronize();

	printf("Q\n");
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			printf("%f ", Q[i][j]);
		}
		printf("\n");
	}
	printf("\n\n");

	printf("R\n");
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			printf("%f ", R[i][j]);
		}
		printf("\n");
	}
	printf("\n\n");

	return 0;
}
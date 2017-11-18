#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int main() {
	int N, K, M;
	// printf("Enter N, K, M: ");
	scanf("%d %d %d", &N, &K, &M);
	float A[N][K];
	float B[K][M];
	float C[N][M];

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < K; j++) {
			scanf("%f", &A[i][j]);
		}
	}

	for (int i = 0; i < K; i++) {
		for (int j = 0; j < M; j++) {
			scanf("%f", &B[i][j]);
		}
	}

	matrix_multiply((float *)A, (float *)B, (float *)C, N, K, M);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			printf("%f ", C[i][j]);
		}
		printf("\n");
	}
	return 0;
}
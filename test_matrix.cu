#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int main() {
	int N, K, M;
	// printf("Enter N, K, M: ");
	scanf("%d", &N);
	scanf("%d", &K);
	scanf("%d", &M);
	float A[N][K];
	float B[K][M];
	float C[N][M];

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < K; j++) {
			scanf("%f", &A[i][j]);
				// printf("HONEYDETECTED\n");
		}
	}

	for (int i = 0; i < K; i++) {
		for (int j = 0; j < M; j++) {
			scanf("%f", &B[i][j]);
		}
	}

	// for (int i = 0; i < N; i++) {
	// 	for (int j = 0; j < K; j++) {
	// 		printf("%f ", A[i][j]);
	// 	}
	// 	printf("\n\n");
	// }

	// for (int i = 0; i < K; i++) {
	// 	for (int j = 0; j < M; j++) {
	// 		printf("%f ", B[i][j]);
	// 	}
	// 	printf("\n\n");
	// }

	matrix_multiply((float *)A, (float *)B, (float *)C, N, K, M);
	cudaDeviceSynchronize();

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			printf("%f ", C[i][j]);
		}
		printf("\n");
	}
	return 0;
}

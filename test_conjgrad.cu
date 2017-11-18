#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "conjgrad.h"

int main() {
	int N;
	// printf("Enter N, K, M: ");
	scanf("%d", &N);
	float A[N][N];
	float b[N];
	float x[N];

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			scanf("%f", &A[i][j]);
		}
	}

	for (int i = 0; i < N; i++) {
		scanf("%f", &b[i]);
	}

	conjgrad((float *)A, x, b, N);
	cudaDeviceSynchronize();

	for (int i = 0; i < N; i++) {
		printf("%f ", x[i]);
	}
	printf("\n");
	return 0;
}
/* Calculates the inverse using gaussian elimination */
#include <stdio.h>
#include <iostream>
#include "ge_gpu.h"
#include "mat_utils.h"
#include "inv_gpu.h"

using namespace std;

int main() {
	int M;
	scanf("%d", &M);
	float A[M][M];

	for (int i=0; i<M; i++) {
		for (int j=0; j<M; j++) {
			scanf("%f", &A[i][j]);
		}
	}

	float inv_A[M][M];

	inverse((float *) A, (float *) inv_A, M);

	if (1) {
		for (int i=0; i<M; i++) {
			for (int j=0; j<M; j++) {
				printf("%f ", inv_A[i][j]);
			}
			printf("\n");
		}
	}
}
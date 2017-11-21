#include <cstdio>
#include <iostream>
#include <cstdlib>
#define true 1
#define false 0

float min_f = -10;
float max_f = 10;

float rand_float(float min=min_f, float max=max_f) {
	return ((float) rand()/RAND_MAX)*(max-min) + min;
}

int main() {
	int M, K, N;

	if (false) {
		scanf("%d", &N);
	}

	// Read square matrix, append rand vector
	if (false) {
		scanf("%d", &N);
		float A[N][N];
		for (int i=0; i<N; i++) {
			for (int j=0; j<N; j++) {
				scanf("%f", &A[i][j]);
			}
		}
		// For ge
		if (false) {
			printf("%d %d\n", N, N+1);
			for (int i=0; i<N; i++) {
				for (int j=0; j<N+1; j++) {
					if (j == N) {
						printf("%f", rand_float());
					} else {
						printf("%f ", A[i][j]);
					}
				}
				printf("\n");
			}
		} else { // For conjgrad
			printf("%d\n", N);
			for (int i=0; i<N; i++) {
				for (int j=0; j<N; j++) {
					printf("%f ", A[i][j]);
				}
				printf("\n");
			}
			for (int i=0; i<N; i++) {
				printf("%f\n", rand_float());
			}
		}
	}

	// Matrix multiplication
	if (false) {
		scanf("%d %d %d", &M, &K, &N);
		printf("%d %d %d\n", M, K, N);
		for (int i=0; i<M; i++) {
			for (int j=0; j<K; j++) {
				printf("%f ", rand_float());
			}
			printf("\n");
		}
		for (int i=0; i<K; i++) {
			for (int j=0; j<N; j++) {
				printf("%f ", rand_float());
			}
			printf("\n");
		}
	}

	// Identity matrix
	if (false) {
		scanf("%d", &N);
		printf("%d\n", N);
		for (int i=0; i<N; i++) {
			for (int j=0; j<N; j++) {
				if (i == j) {
					printf("1 ");
				} else {
					printf("0 ");
				}
			}
			printf("\n");
		}
	}

	// Identity matrix with (N, N) instead of N
	if (true) {
		scanf("%d", &N);
		printf("%d %d\n", N, N);
		for (int i=0; i<N; i++) {
			for (int j=0; j<N; j++) {
				if (i == j) {
					printf("1 ");
				} else {
					printf("0 ");
				}
			}
			printf("\n");
		}
	}

	// vector of ones
	if (false) {
		for (int i=0; i<N; i++) {
			printf("1\n");
		}
	}

	// identity matrix | vector of ones
	if (false) {
		printf("%d %d\n", N, N+1);
		for (int i=0; i<N; i++) {
			for (int j=0; j<N+1; j++) {
				if (j == N) {
					printf("1");
					continue;
				}
				if (i == j) {
					printf("1 ");
				} else {
					printf("0 ");
				}
			}
			printf("\n");
		}
	}
}
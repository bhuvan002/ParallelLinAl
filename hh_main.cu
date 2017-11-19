/* Householders method to calculate the QR decomposition */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "mat_utils.h"
#include "givens.h"

void hh_gpu(float *A, int M, int N, float *Q, float *R) {

	for (int i=0; i<M; i++) {
		for (int i=0; i<M; i++) {
			Q[idx(i,i,M)] = 0;
		}
	}
	for (int i=0; i<M; i++) {
		Q[idx(i,i,M)] = 1;
	}
	memcpy(R, A, M*N*sizeof(float));

	float normx, u1, tau;
	int s;

	for (int j=0; j<N; j++) {
		normx = 0;
		for (int i=j; i<M; i++) {
			int num = R[idx(i,j,N)];
			normx += num*num;
		}
		normx = sqrt(normx);
		s = -((R[idx(j,j,N)]<0)?-1:1);
		u1 = R[idx(j,j,N)] - s*normx;

		float w[M-j];
		for (int i=j; i<M; i++) {
			w[i-j] = R[idx(i,j,N)]/u1;
		}
		w[0] = 1;
		tau = -s*u1/normx;
		// R = HR, Q = QH
		// H = I - tau * w * w'
		float R_sub[M-j][N];
		float w_t_R[N];
		float tau_w[M-j];
		float R_rhs[M-j][N];
		float Q_sub[M][M-j];
		float Q_w[M][1];
		float Q_rhs[M][M-j];

		copy_sub_matrix((float *) R, (float *) R_sub, M, N, j, 0);
		copy_sub_matrix((float *) Q, (float *) Q_sub, M, M, 0, j);

		if (1) {
			givens_rotation(A, Q, R, M, N);
			return;
		}

		// R(j:end,:) = R(j:end,:)-(tau*w)*(w’*R(j:end,:));
		matrix_multiply(w, (float *) R_sub, (float *) w_t_R, 1, M-j, N);
		scalar_mul(w, tau_w, tau, M-j, 1);
		matrix_multiply(tau_w, w_t_R, (float *) R_rhs, M-j, 1, N);
		matrix_sub((float *) R_sub, (float *) R_rhs, (float *) R_sub, M-j, N);
		copy_back_sub_matrix((float *) R, (float *) R_sub, M, N, j, 0, M-1, N-1);

		// Q(:,j:end) = Q(:,j:end)-(Q(:,j:end)*w)*(tau*w)’;
		matrix_multiply((float *) Q_sub, w, (float *) Q_w, M, M-j, 1);
		matrix_multiply((float *) Q_w, tau_w, (float *) Q_rhs, M, 1, M-j);
		matrix_sub((float *) Q_sub, (float *) Q_rhs, (float *) Q_sub, M, M-j);
		copy_back_sub_matrix((float *) Q, (float *) Q_sub, M, M, 0, j, M-1, M-1);
	}
}

int main() {
	int M, N;
	scanf("%d %d", &M, &N);
	float A[M][N];
	for (int i=0; i<M; i++) {
		for (int j=0; j<N; j++) {
			float num;
			scanf("%f", &num);
			A[i][j] = num;
		}
	}

	float Q[M][M];
	float R[M][N];
	hh_gpu((float *) A, M, N, (float *) Q, (float *) R);

	if (1) {
		printf("Q:\n");
		for (int i=0; i<M; i++) {
			for (int j=0; j<M; j++) {
				printf("%f ", Q[i][j]);
			}
			printf("\n");
		}

		printf("R:\n");
		for (int i=0; i<M; i++) {
			for (int j=0; j<N; j++) {
				printf("%f ", R[i][j]);
			}
			printf("\n");
		}
	}
}
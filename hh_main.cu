/* Householders method to calculate the QR decomposition */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

void copy_sub_matrix(float *A, float *sub_A, int M, int N, int r, int c) {
	int sub_M = M-r;
	int sub_N = N-c;

	for (int i=r; i<M; i++) {
		for (int j=c; j<N; j++) {
			sub_A[idx(i-r,j-c,sub_N)] = A[idx(i,j,N)];
		}
	}
}

void copy_back_sub_matrix(float *A, float *sub_A, int M, int N, int r, int c) {
	int sub_M = M-r;
	int sub_N = N-c;

	for (int i=r; i<M; i++) {
		for (int j=c; j<N; j++) {
			A[idx(i,j,N)] = sub_A[idx(i-r,j-c,sub_N)];
		}
	}
}

void scalar_mul(float *A, float *scaled_A, int scalar, int M, int N) {
	for (int i=0; i<M; i++) {
		for (int j=0; j<N; j++) {
			scaled_A[idx(i,j,N)] = A[idx(i,j,N)] * scalar;
		}
	}
}

void hh_cpu(float *A, int M, int N, float *Q, float *R) {
	for (int i=0; i<M; i++) {
		Q[i][i] = 0;
	}
	memcpy(R, A, M*N*sizeof(float));

	float normx, u1, tau;
	int s;
	float w[M]; // M-j+1

	for (int j=0; j<N; j++) {
		normx = 0
		for (int i=j; i<M; i++) {
			normx += R[i][j]*R[i][j];
		}
		normx = sqrt(normx);
		s = -((R[j][j]<0)?-1:1);
		u1 = R[j][j] - s*normx;

		float w[M-j];
		for (int i=j; i<M; i++) {
			w[i-j] = R[i][j]/u1;
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

		copy_sub_matrix(R, R_sub, M, N, j, 0);
		copy_sub_matrix(Q, Q_sub, M, M, 0, j);

		// R(j:end,:) = R(j:end,:)-(tau*w)*(w’*R(j:end,:));
		matrix_multiply(w, (float *) R_sub, (float *) w_t_R, 1, M-j, N);
		scalar_mul(w, tau_w, tau, M-j, 1);
		matrix_multiply(tau_w, w_t_R, R_rhs, M-j, 1, N);
		matrix_sub(R_sub, R_rhs, R_sub, M-j, N);
		copy_back_sub_matrix(R, R_sub, M, N, j, 0);

		// Q(:,j:end) = Q(:,j:end)-(Q(:,j:end)*w)*(tau*w)’;
		matrix_multiply((float *) Q_sub, w, (float *) Q_w, M, M-j, 1);
		matrix_multiply((float *) Q_w, taw_w, (float *) Q_rhs, M, 1, M-j);
		matrix_sub(Q_sub, Q_rhs, Q_sub, M, M-j);
		copy_back_sub_matrix(Q, Q_sub, M, M, 0, j);
	}
}

void main() {
	
}
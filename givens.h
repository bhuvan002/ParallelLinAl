__host__ __device__
void givens(float a, float b, float *c, float *s, float *r);
void givens_rotation(float *A, float *Q, float *R, int M, int N);
__global__ void givens_rotate_R(float *R_t, int M, int N, int i, int rot_col, 
		float c, float s, float r);
__global__ void givens_rotate_Q(float *Q, int M, int N, int j, float c, float s);
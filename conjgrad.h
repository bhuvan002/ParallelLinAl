struct saxpy_functor {

	const float a;

	saxpy_functor(float _a) : a(_a) {}

	__host__ __device__
	float operator()(const float &x, const float &y) {
		return a * x + y;
	}
};


__global__ void zeros(float *A, int N);
void conjgrad(float *A, float *x, float *b, int N);

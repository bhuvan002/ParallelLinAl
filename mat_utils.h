#ifndef MAT_UTILS_H
#define MAT_UTILS_H

__device__ __host__
inline int idx(int x, int y, int num_cols) {
	return x*num_cols+y;
}

__device__ __host__ 
inline int num_blocks(int tot_threads, int t_per_b) {
	return (tot_threads+t_per_b-1)/t_per_b;
}

#endif
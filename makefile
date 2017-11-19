CFLAGS=-g

conjgrad: test_conjgrad.o matrix.o conjgrad.o
	nvcc $(CFLAGS) test_conjgrad.o matrix.o conjgrad.o -o conjgrad

matrix: test_matrix.o matrix.o
	nvcc $(CFLAGS) test_matrix.o matrix.o -o matrix

ge_main: ge_main.o matrix.o mat_utils.o
	nvcc $(CFLAGS) ge_main.o matrix.o mat_utils.o -o ge_main

hh_main: hh_main.o matrix.o mat_utils.o
	nvcc $(CFLAGS) hh_main.o matrix.o mat_utils.o -o hh_main

inv_main: inv_main.o matrix.o ge_gpu.o mat_utils.o inv_gpu.o
	nvcc $(CFLAGS) inv_main.o matrix.o ge_gpu.o mat_utils.o inv_gpu.o -o inv_main

inv_gpu.o: inv_gpu.cu
	nvcc $(CFLAGS) -c inv_gpu.cu

givens: test_givens.o matrix.o givens.o
	nvcc $(CFLAGS) test_givens.o matrix.o givens.o -o givens

givens.o: givens.cu
	nvcc $(CFLAGS) -c givens.cu

test_givens.o: test_givens.cu
	nvcc $(CFLAGS) -c test_givens.cu

ge_main: ge_main.o matrix.o
	nvcc ge_main.o matrix.o

ge_main.o: ge_main.cu
	nvcc $(CFLAGS) -c ge_main.cu

ge_gpu.o: ge_gpu.cu
	nvcc $(CFLAGS) -c ge_gpu.cu

hh_main.o: hh_main.cu
	nvcc $(CFLAGS) -c hh_main.cu

inv_main.o: inv_main.cu
	nvcc $(CFLAGS) -c inv_main.cu

mat_utils.o: mat_utils.cu
	nvcc $(CFLAGS) -c mat_utils.cu

matrix.o: matrix.cu
	nvcc $(CFLAGS) -c matrix.cu

test_matrix.o: test_matrix.cu
	nvcc $(CFLAGS) -c test_matrix.cu

test_conjgrad.o: test_conjgrad.cu
	nvcc $(CFLAGS) -c test_conjgrad.cu

conjgrad.o: conjgrad.cu
	nvcc $(CFLAGS) -c conjgrad.cu

.PHONY: clean

clean:
	rm -rf *.o
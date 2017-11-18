CFLAGS=-g

conjgrad: test_conjgrad.o matrix.o conjgrad.o
	nvcc $(CFLAGS) test_conjgrad.o matrix.o conjgrad.o -o conjgrad

matrix: test_matrix.o matrix.o
	nvcc $(CFLAGS) test_matrix.o matrix.o -o matrix

ge_main: ge_main.o matrix.o
	nvcc ge_main.o matrix.o

ge_main.o: ge_main.cu
	nvcc -c ge_main.cu

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
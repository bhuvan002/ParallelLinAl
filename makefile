all:	a.out

a.out: test_matrix.o matrix.o
	nvcc test_matrix.o matrix.o

matrix.o: matrix.cu
	nvcc -c matrix.cu

test_matrix.o: test_matrix.cu
	nvcc -c test_matrix.cu

.PHONY: clean

clean:
	rm -rf *.o a.out
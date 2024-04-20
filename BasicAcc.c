#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void matrixMultiplication(int *matrixA, int *matrixB, int *matrixC, int rowsA, int colsA, int colsB) {
    #pragma acc parallel loop collapse(3) present(matrixA[:rowsA*colsA], matrixB[:colsA*colsB], matrixC[:rowsA*colsB])
    for (int i=0; i<rowsA; i++) {
        for (int j=0; j<colsB; j++) {
            int sum=0;
            for (int k=0; k<colsA; k++) {
                sum += matrixA[i*colsA+k] * matrixB[k*colsB+j];
            }
            matrixC[i*colsB+j] = sum;
        }
    }
}

int main() {
	int *matrixA, *matrixB, *matrixC;
    int rowsA, colsA, colsB;
    printf("Enter number of rows and columns of A: ");
    scanf("%d %d", &rowsA, &colsA);
    printf("Enter number of columns of B: "); //rows of B must be equal to columns of A
    scanf("%d", &colsB);
    matrixA = (int*)malloc(rowsA*colsA*sizeof(int));
    matrixB = (int*)malloc(colsA*colsB*sizeof(int));
    matrixC = (int*)malloc(rowsA*colsB*sizeof(int));
    for (int i=0; i<rowsA*colsA; i++) {
        matrixA[i] = rand() % 10;
    }
    for (int i=0; i<colsA*colsB; i++) {
        matrixB[i] = rand() % 10;
    }
	double start = omp_get_wtime();
    matrixMultiplication(matrixA, matrixB, matrixC, rowsA, colsA, colsB);
    double end = omp_get_wtime();
    printf("Time: %f ms\n", (end - start)*1000);
    free(matrixA);
    free(matrixB);
    free(matrixC);
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_WIDTH 16

void matrixMultiplication(int* A, int* B, int* C, int rowsA, int colsA, int colsB) {
    int bx, by, tx, ty, Row, Col;
    float Cvalue;

    #pragma acc data copyin(A[0:rowsA*colsA], B[0:colsA*colsB]) copy(C[0:rowsA*colsB])
    {
        #pragma acc kernels
        #pragma acc loop tile(TILE_WIDTH, TILE_WIDTH)
        for (by=0; by<rowsA; by+=TILE_WIDTH) {
            for (bx=0; bx<colsB; bx+=TILE_WIDTH) {
                for (ty=0; ty<TILE_WIDTH; ty++) {
                    for (tx=0; tx<TILE_WIDTH; tx++) {
                        Row = by + ty;
                        Col = bx + tx;
                        if (Row<rowsA && Col<colsB) {
                            Cvalue = 0.0;
                            for (int k=0; k<colsA; k++) {
                                Cvalue += A[Row*colsA+k] * B[k*colsB+Col];
                            }
                            C[Row*colsB+Col] = Cvalue;
                        }
                    }
                }
            }
        }
    }
}

int main() {
    int rowsA, colsA, colsB;
    printf("Enter number of rows and columns of A: ");
    scanf("%d %d", &rowsA, &colsA);
    printf("Enter number of columns of B: ");
    scanf("%d", &colsB);

    int *matrixA = (int *)malloc(rowsA * colsA * sizeof(int));
    int *matrixB = (int *)malloc(colsA * colsB * sizeof(int));
    int *matrixC = (int *)malloc(rowsA * colsB * sizeof(int));

    for (int i = 0; i < rowsA * colsA; i++) {
        matrixA[i] = rand() % 10;
    }

    for (int i = 0; i < colsA * colsB; i++) {
        matrixB[i] = rand() % 10;
    }

    // Perform matrix multiplication
    clock_t start = clock();
    matrixMultiplication(matrixA, matrixB, matrixC, rowsA, colsA, colsB);
    clock_t end = clock();

    // Calculate elapsed time
    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %f ms\n", elapsed_time*1000);

    // Cleanup
    free(matrixA);
    free(matrixB);
    free(matrixC);
    return 0;
}

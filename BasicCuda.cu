#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define BLOCK_SIZE 16

__global__ void matrixMultiplication(int *matrixA, int *matrixB, int *matrixC, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (row < rowsA && col < colsB) {
        for (int k=0; k<colsA; k++) {
            sum += matrixA[row*colsA+k] * matrixB[k*colsB+col];
        }
        matrixC[row*colsB+col] = sum;
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
    int *d_matrixA, *d_matrixB, *d_matrixC;
    cudaMalloc(&d_matrixA, rowsA * colsA * sizeof(int));
    cudaMalloc(&d_matrixB, colsA * colsB * sizeof(int));
    cudaMalloc(&d_matrixC, rowsA * colsB * sizeof(int));
    cudaMemcpy(d_matrixA, matrixA, rowsA * colsA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixB, matrixB, colsA * colsB * sizeof(int), cudaMemcpyHostToDevice);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((colsB + blockDim.x - 1) / blockDim.x, (rowsA + blockDim.y - 1) / blockDim.y);
    clock_t start = clock();
    matrixMultiplication<<<gridDim, blockDim>>>(d_matrixA, d_matrixB, d_matrixC, rowsA, colsA, colsB);
    clock_t end = clock();
    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %f ms\n", elapsed_time*1000);
    cudaMemcpy(matrixC, d_matrixC, rowsA * colsB * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_matrixA);
    cudaFree(d_matrixB);
    cudaFree(d_matrixC);
    free(matrixA);
    free(matrixB);
    free(matrixC);
    return 0;
}

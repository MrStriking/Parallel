#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define TILE_WIDTH 16
#define BLOCK_SIZE 16

__global__ void matrixMultiplication(int* A, int* B, int* C, int rowsA, int colsA, int colsB) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    float Cvalue= 0;
    for (int p = 0; p < colsA/TILE_WIDTH; ++p) {
      ds_A[ty][tx] = A[Row*colsB + p*TILE_WIDTH+tx];
      ds_B[ty][tx] = B[(p*TILE_WIDTH+ty)*colsB + Col];
      __syncthreads();
      for (int i = 0; i < TILE_WIDTH; ++i) 
        Cvalue+= ds_A[ty][i] * ds_B[i][tx];
      __syncthreads();
    }
    C[Row*colsB+Col] = Cvalue;
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

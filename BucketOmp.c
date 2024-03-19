#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAX_VALUE 1000
#define N 100
 
void bucketSort(int A[], int size) {  
    int bucket[MAX_VALUE+1] = {0};
    int B[size];
    #pragma omp parallel for
    for (int i=0; i<size; i++) {
        #pragma omp atomic
        bucket[A[i]]++;
    }
    int sum = 0;
    for (int i=0; i<=MAX_VALUE; i++) {
        int count = bucket[i];
        bucket[i] = sum;
        sum += count;
    }
    #pragma omp parallel for
    for (int i=0; i<size; i++) {
        int value = A[i];
        int index;
        #pragma omp atomic capture 
        {
            index = bucket[value];
            bucket[value]++;
        }
        B[index] = value;
    }
    #pragma omp parallel for
    for (int i=0; i<size; i++) {
        A[i] = B[i];
    }  
}  
 
int main() {
	int *A = malloc(N * sizeof(int));
    for (int i=0; i<N; i++) {
        A[i] = rand() % MAX_VALUE;
    }   
    double start = omp_get_wtime();
    bucketSort(A, N); 
    double end = omp_get_wtime();
    double elapsed_time = end - start;
    for (int i=0; i<N; i++) {
        printf("%d ", A[i]);
    }   
    printf("\nElapsed time: %.5f ms\n", elapsed_time*1000);    
    return 0;
}

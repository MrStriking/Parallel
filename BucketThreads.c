#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define MAX_VALUE 1000
#define NUM_THREADS 4
#define N 100 

typedef struct {
    int* arr;
    int size;
    int start;
    int end;
    int* bucket;
} Thread;

void* Range(void* arg) {
    Thread* thread = (Thread*)arg;
    for (int i=thread->start; i<thread->end; i++) {
        thread->bucket[thread->arr[i]]++;
    }
    return NULL;
}

void bucketSort(int A[], int size) {
    pthread_t threads[NUM_THREADS];
    Thread threadStruct[NUM_THREADS];
    int* bucket = (int*)calloc(MAX_VALUE + 1, sizeof(int));
    int chunkSize = size / NUM_THREADS;
    for (int i=0; i<NUM_THREADS; i++) {
        threadStruct[i].arr = A;
        threadStruct[i].size = size;
        threadStruct[i].start = i * chunkSize;
        if (i==NUM_THREADS-1) {
        	threadStruct[i].end = size;
        } else {
        	threadStruct[i].end = (i + 1) * chunkSize;
        }
        threadStruct[i].bucket = bucket;
        pthread_create(&threads[i], NULL, Range, &threadStruct[i]);
    }
    for (int i=0; i<NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    int index=0;
    for (int i=0; i<=MAX_VALUE; i++) {
        for (int j=0; j<bucket[i]; j++) {
            A[index++] = i;
        }
    }
    free(bucket);
} 
 
int main() {
	int *A = malloc(N * sizeof(int));
    for (int i=0; i<N; i++) {
        A[i] = rand() % MAX_VALUE;
    }   
    clock_t start = clock();
    bucketSort(A, N); 
    clock_t end = clock();
    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC; 
    for (int i=0; i<N; i++) {
        printf("%d ", A[i]);
    }   
    printf("\nElapsed time: %.5f ms\n", elapsed_time*1000);    
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

void cal_pixel(int start, int end, int* area) {
    int index = 0;
    for (int i=start; i < end; i++) {
        for (int j=0; j < WIDTH; j++) {
            double x = (j - WIDTH / 2.0) * 4.0 / WIDTH;
            double y = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
            double real = x;
            double imag = y;
            int k;
            for (k = 0; k < MAX_ITER; k++) {
                double real2 = real * real;
                double imag2 = imag * imag;
                if (real2 + imag2 > 4.0)
                    break;
                imag = 2*real*imag + y;
                real = real2-imag2 + x;
            }
            area[index++] = k;
        }
    }
}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) { //copied from the provided sequential code
    FILE* pgmimg; 
    int temp;
    pgmimg = fopen(filename, "wb"); 
    fprintf(pgmimg, "P2\n"); 
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);
    fprintf(pgmimg, "255\n");
    
    for (int i = 0; i < HEIGHT; i++) { 
        for (int j = 0; j < WIDTH; j++) { 
            temp = image[i][j]; 
            fprintf(pgmimg, "%d ", temp);
        } 
        fprintf(pgmimg, "\n"); 
    } 
    fclose(pgmimg); 
}

int main(int argc, char *argv[]) {
    int image[HEIGHT][WIDTH];
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rfp = HEIGHT / size;
    int start = rank * rfp;
    int end = (rank+1) * rfp;
    int area_size = rfp * WIDTH;
    int area[area_size];
    double start_time = MPI_Wtime(); 



  
    double end_time = MPI_Wtime();
    if (rank == 0) {  
        printf("Execution time: %.3f ms\n", (end_time - start_time)*1000);
        save_pgm("mandelbrotdynamic.pgm", image);
    }
    MPI_Finalize();
    return 0;
}

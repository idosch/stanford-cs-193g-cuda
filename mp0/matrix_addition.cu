/* CUDA matrix addition (C = A + B) where each thread is responsilbe for one
 * element in matrix C
*/
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>

/* Generate two random matrices of dimension nxn with float precision. */
void matGen(float*, float*, int);

/* Adds two matrices of dimension nxn. */
void matAdd(float*, float*, float*, int);

/* Prints matrices - for debug. */
void printMat(float*, float*, float*, int);

/* Device matrix addition. Each thread performs one pair-wise addition. */
__global__
void matAddKernel(float*, float*, float*, int n);

int main(int argc, char* argv[]) {
    float *A, *B, *C;       /* Matrices */
    const int n = 1<<10;    /* Matrices dimension (square) */

    A = (float*) malloc(n*n*sizeof(float));
    B = (float*) malloc(n*n*sizeof(float));
    C = (float*) malloc(n*n*sizeof(float));

    /* Generate A and B */
    matGen(A, B, n);
    /* Compute C = A + B */
    matAdd(A, B, C, n);
    /* Print matrices - for debug */
    //printMat(A, B, C, n);

    return EXIT_SUCCESS;
}

void matGen(float* A, float* B, int n) {
    int i;

    for (i = 0; i < n*n; i++) {
        A[i] = (float)rand() / (float)RAND_MAX;
        B[i] = (float)rand() / (float)RAND_MAX;
    }
}

void matAdd(float* A, float* B, float* C, int n) {
    float *d_A, *d_B, *d_C;    /* Matrices A, B and C on the device */
    int size;                   /* Number of bytes to allocate */
   
    /* Allocated space for the matrices on the devices - no error checking in
    order to maintain readability */
    size = n*n*sizeof(float);
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    /* Transfer input data to the device */
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    /* Kernel execution */
    matAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);

    /* Get results back from device and do cleanup */
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

__global__
void matAddKernel(float* A, float* B, float* C, int n) {
    int i;

    i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n*n) C[i] = A[i] + B[i];
}

void printMat(float* A, float* B, float* C, int n) {
    int i, j;

    printf("Matrix A:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", A[i*n + j]);
        }
        printf("\n");
    }

    printf("Matrix B:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", B[i*n + j]);
        }
        printf("\n");
    }

    printf("Matrix C:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", C[i*n + j]);
        }
        printf("\n");
    }
}

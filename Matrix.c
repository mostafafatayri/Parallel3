#include <stdio.h>
#include <cuda.h>

#define BLOCK_SIZE 16

__global__ void matrixMulBasic(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}


int main(int argc, char *argv[]) {
    int M = 10024, N = 10024, K = 10024;
    float *h_A = (float *)malloc(M * K * sizeof(float));
    float *h_B = (float *)malloc(K * N * sizeof(float));
    float *h_C = (float *)malloc(M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) {
        h_A[i] = rand() % 100;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = rand() % 100;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    // Copy host data to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    matrixMulBasic<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    printf("Elapsed time: %.2f ms\n", milliseconds);

    return 0;
}

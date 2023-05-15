#include <stdio.h>
#include <cuda.h>

#define BLOCK_SIZE 16

__global__ void matrixMulTiling(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0;

    for (int tileIdx = 0; tileIdx < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; tileIdx++) {
        if (row < M && tileIdx * BLOCK_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + tileIdx * BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0;

        if (col < N && tileIdx * BLOCK_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(tileIdx * BLOCK_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
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

 
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

   
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    matrixMulTiling<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Elapsed time: %f ms\n", elapsedTime);
}


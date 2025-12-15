#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

#define TILE_SIZE 32
#define N 2048  // Matrix size 2048 x 2048

// Error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// CUDA Kernel: Tiled Matrix Multiplication
// C = A * B
__global__ void matrixMulTiled(const float* A, const float* B, float* C, int width) {
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the C element to work on
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float val = 0.0f;

    // Loop over the tiles of input matrices required to compute the C element
    for (int m = 0; m < width / TILE_SIZE; ++m) {
        
        // Shared memory for the sub-matrices (tiles)
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        // Load the tiles from global memory into shared memory
        // A is accessed row-wise, B is accessed column-wise
        As[ty][tx] = A[row * width + (m * TILE_SIZE + tx)];
        Bs[ty][tx] = B[(m * TILE_SIZE + ty) * width + col];

        // Ensure all threads have loaded the tile before computing
        __syncthreads();

        // Multiply the two tiles together
        for (int k = 0; k < TILE_SIZE; ++k) {
            val += As[ty][k] * Bs[k][tx];
        }

        // Ensure computation is done before loading the next tile
        __syncthreads();
    }

    // Write the result to global memory
    if (row < width && col < width) {
        C[row * width + col] = val;
    }
}

int main() {
    // 1. Setup size and host memory
    size_t bytes = N * N * sizeof(float);
    std::cout << "Matrix Size: " << N << " x " << N << std::endl;

    float *h_A, *h_B, *h_C, *h_D;
    
    // Use cudaMallocHost for pinned memory (faster transfers)
    cudaCheckError(cudaMallocHost((void**)&h_A, bytes));
    cudaCheckError(cudaMallocHost((void**)&h_B, bytes));
    cudaCheckError(cudaMallocHost((void**)&h_C, bytes));
    cudaCheckError(cudaMallocHost((void**)&h_D, bytes));

    // Initialize with random values
    std::cout << "Initializing host matrices..." << std::endl;
    for (int i = 0; i < N * N; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        h_C[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 2. Allocate device memory
    float *d_A, *d_B, *d_C, *d_Temp, *d_D;
    cudaCheckError(cudaMalloc((void**)&d_A, bytes));
    cudaCheckError(cudaMalloc((void**)&d_B, bytes));
    cudaCheckError(cudaMalloc((void**)&d_C, bytes));
    cudaCheckError(cudaMalloc((void**)&d_Temp, bytes)); // Intermediate buffer
    cudaCheckError(cudaMalloc((void**)&d_D, bytes));    // Final Result

    // 3. Copy Host to Device
    std::cout << "Copying data to GPU..." << std::endl;
    cudaCheckError(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice));

    // 4. Setup Execution Configuration
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Launching kernels..." << std::endl;
    cudaEventRecord(start);

    // =========================================================
    // SEQUENTIAL LAUNCH
    // =========================================================
    
    // Kernel 1: Temp = A * B
    matrixMulTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Temp, N);
    
    // Kernel 2: D = Temp * C
    matrixMulTiled<<<blocksPerGrid, threadsPerBlock>>>(d_Temp, d_C, d_D, N);

    // =========================================================

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Check for kernel errors
    cudaCheckError(cudaGetLastError());

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Computation Complete. Time: " << milliseconds << " ms" << std::endl;

    // 5. Copy Result back to Host
    cudaCheckError(cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost));

    // Verification (Corner check)
    std::cout << "D[0] = " << h_D[0] << std::endl;
    std::cout << "D[last] = " << h_D[N*N - 1] << std::endl;

    // 6. Free Memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_Temp); cudaFree(d_D);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C); cudaFreeHost(h_D);

    return 0;
}
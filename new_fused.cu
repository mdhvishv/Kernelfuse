#include <cuda_runtime.h>
#include <stdio.h>

#define N 2048
#define TILE_SIZE 32

// ----------------------------------------------------------------------
// ATOMIC FUSED KERNEL
// Grid Dimensions: Mapped to the INTERMEDIATE matrix (A*B).
// 1. Compute one tile of (A*B).
// 2. Iterate across C to multiply that tile against C.
// 3. Atomically add the result to D in global memory.
// ----------------------------------------------------------------------
__global__ void atomicFusedMatrixMultiply(const float* __restrict__ A, 
                                          const float* __restrict__ B,
                                          const float* __restrict__ C,
                                          float* __restrict__ D) {
    // Shared memory for inputs
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];
    
    // Shared memory for the Computed Intermediate Tile (A*B)
    __shared__ float s_AB[TILE_SIZE][TILE_SIZE];
    
    // Shared memory for C
    __shared__ float s_C[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int tx = threadIdx.x; 
    int ty = threadIdx.y;

    // -------------------------------------------------------
    // PHASE 1: Compute ONE tile of AB (The Intermediate)
    // -------------------------------------------------------
    
    float acc_AB = 0.0f;
    
    // Standard Loop to compute A * B for this specific tile
    for (int k = 0; k < N / TILE_SIZE; ++k) {
        int rowA = by * TILE_SIZE + ty;
        int colA = k * TILE_SIZE + tx;
        s_A[ty][tx] = A[rowA * N + colA];

        int rowB = k * TILE_SIZE + ty;
        int colB = bx * TILE_SIZE + tx;
        s_B[ty][tx] = B[rowB * N + colB];

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            acc_AB += s_A[ty][i] * s_B[i][tx];
        }
        __syncthreads();
    }
    
    // Store calculated AB tile in shared memory
    s_AB[ty][tx] = acc_AB;
    __syncthreads();

    // -------------------------------------------------------
    // PHASE 2: Sweep across C and Atomically Add to D
    // -------------------------------------------------------
    
    int row_AB = ty; 
    // REMOVED: int col_AB = tx; (This was the unused variable)
    
    // Iterate over all column blocks of C
    for (int c_blk = 0; c_blk < N / TILE_SIZE; ++c_blk) {
        
        // Load tile from C
        int rowC = bx * TILE_SIZE + ty; 
        int colC = c_blk * TILE_SIZE + tx;
        
        s_C[ty][tx] = C[rowC * N + colC];
        __syncthreads();
        
        // Compute partial dot product
        float partial_D = 0.0f;
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            partial_D += s_AB[row_AB][k] * s_C[k][tx];
        }
        
        // ---------------------------------------------------
        // PHASE 3: Atomic Add to Global Memory
        // ---------------------------------------------------
        int global_D_Row = by * TILE_SIZE + ty;
        int global_D_Col = c_blk * TILE_SIZE + tx;
        
        atomicAdd(&D[global_D_Row * N + global_D_Col], partial_D);
        
        __syncthreads();
    }
}
int main() {
    size_t bytes = N * N * sizeof(float);
    
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_D = (float*)malloc(bytes);
    
    // Initialize randoms
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
        h_B[i] = (float)(rand() % 100) / 100.0f;
        h_C[i] = (float)(rand() % 100) / 100.0f;
        h_D[i] = 0.0f; // IMPORTANT: D must be 0 for atomicAdd to work
    }
    
    float *d_A, *d_B, *d_C, *d_D;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMalloc(&d_D, bytes);
    
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice);
    // Initialize D to 0 on device
    cudaMemset(d_D, 0, bytes);
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("Launching ATOMIC FUSED kernel for N = %d...\n", N);
    
    // Note: Grid dimensions match A and B (to compute AB tiles)
    atomicFusedMatrixMultiply<<<blocks, threads>>>(d_A, d_B, d_C, d_D);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost);
    
    printf("Completed. Top left element: %f\n", h_D[0]);
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    free(h_A); free(h_B); free(h_C); free(h_D);
    
    return 0;
}
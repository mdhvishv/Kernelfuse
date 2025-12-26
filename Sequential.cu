#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>

// === Tiling Parameters ===
// BM/BN: Block size in M and N dimensions (Output tile size per block)
// BK: Block size in K dimension (Step size for dot product loop)
// TM/TN: Tile size in M and N (Output tile size per thread)
#define BM 64
#define BN 64
#define BK 8
#define TM 4
#define TN 4

#define N 4096 

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Optimized Kernel with 2D Register Tiling
__global__ void matrixMulRegisterTiled(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int width) {
    // 1. Thread Indexing
    // We treat the block of threads as a 2D grid that maps to the output tile
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    
    // Thread index within the block (0-255)
    // We map this linearly to handle loading shared memory
    const uint tid = threadIdx.y * blockDim.x + threadIdx.x;

    // 2. Allocate Shared Memory
    // These tile sizes are static for performance
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // 3. Allocate Registers (Thread Local)
    // This is the "Register Tile" - 4x4 accumulator per thread
    float threadResults[TM][TN] = {0.0f};

    // Registers to cache values from Shared Memory during computation
    float regM[TM];
    float regN[TN];

    // Calculate the Global Row and Column this thread works on
    // stride is blockDim.x (16) * blockDim.y (16) = 256 threads
    // But we are mapping to a 64x64 tile.
    // Threads are arranged 16x16. Each thread handles a 4x4 patch.
    uint threadRow = threadIdx.y; // 0..15
    uint threadCol = threadIdx.x; // 0..15

    // Advance pointers to the starting position for this block
    const float* A_ptr = A + (cRow * BM * width);
    const float* B_ptr = B + (cCol * BN);
    float* C_ptr = C + (cRow * BM * width) + (cCol * BN);

    // 4. Main Loop over K dimension
    for (uint bkIdx = 0; bkIdx < width; bkIdx += BK) {
        
        // --- Collaborative Loading into Shared Memory ---
        // We need to load BM x BK (64x8 = 512 floats) and BK x BN (8x64 = 512 floats).
        // We have 256 threads. Each thread loads 2 floats for A and 2 floats for B.
        
        // Load A (Row-Major)
        // Map tid (0..255) to As indices. 
        // 512 elements total. Thread i loads element i and i+256?
        // Let's use a simpler strided load for clarity:
        
        // Load As: Matrix A is [BM][BK] in shared.
        // Each thread loads specific elements.
        // Using int4/float4 here would be the next optimization step.
        // For now, manual unrolling for generic float:
        
        // Load As (Dimensions BM x BK)
        // We reinterpret 256 threads to load 64*8 = 512 elements. 
        // Each thread loads 2 elements.
        int innerRowA = tid / BK; // 0..63
        int innerColA = tid % BK; // 0..7
        int strideA   = (blockDim.x * blockDim.y) / BK; // 256/8 = 32 rows stride
        
        // This loading logic is specific to 256 threads loading 512 elements
        // This is a "vanilla" load for stability. Vectorized load is faster.
        if (tid < 512) { // Guard if block size changes
             // A is stored row-major. 
             // Global Memory Index: (Row * width) + Col
             // We need to handle the offset `bkIdx` for the columns of A
             // But actually, for 100% coalescing, we usually load A transposed or use complex indexing.
             // Let's stick to a simpler mapping:
             // Each thread loads:
             // As[row][col]
             // We map the 256 threads to cover the 64x8 area.
             // Row = tid / 8, Col = tid % 8. 
             // Since 256 < 512, each thread loops twice?
             // Let's just use the linear TID loop:
             for(int loadOffset = 0; loadOffset < (BM*BK)/(256); loadOffset++) {
                 int idx = tid + loadOffset * 256;
                 int r = idx / BK;
                 int c = idx % BK;
                 As[r][c] = A_ptr[r * width + c + bkIdx]; 
             }
        }

        // Load Bs (Dimensions BK x BN)
        // 8x64 = 512 elements.
        if (tid < 512) {
             for(int loadOffset = 0; loadOffset < (BK*BN)/(256); loadOffset++) {
                 int idx = tid + loadOffset * 256;
                 int r = idx / BN;
                 int c = idx % BN;
                 Bs[r][c] = B_ptr[(r + bkIdx) * width + c];
             }
        }

        __syncthreads();

        // --- Computation (Register Tiled) ---
        // Iterate over the K dimension of the block (0..BK)
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            
            // 1. Prefetch rows of A and cols of B from Shared into Registers
            // We need A[threadRow*TM ... +TM][dotIdx]
            // We need B[dotIdx][threadCol*TN ... +TN]
            
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[threadRow * TM + i][dotIdx];
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx][threadCol * TN + i];
            }

            // 2. Compute Outer Product (TM x TN)
            // This loop is purely register-math (Very Fast)
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM][resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // 5. Write Results to Global Memory
    // Each thread writes its 4x4 sub-matrix
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
             int row = (cRow * BM) + (threadRow * TM) + resIdxM;
             int col = (cCol * BN) + (threadCol * TN) + resIdxN;
             if (row < width && col < width) {
                 C[row * width + col] = threadResults[resIdxM][resIdxN];
             }
        }
    }
}

int main() {
    // 1. Setup
    size_t bytes = N * N * sizeof(float);
    std::cout << "Matrix Size: " << N << " x " << N << std::endl;
    std::cout << "Optimization: 2D Register Tiling (Thread coarsening)" << std::endl;

    float *h_A, *h_B, *h_C, *h_D;
    cudaCheckError(cudaMallocHost((void**)&h_A, bytes));
    cudaCheckError(cudaMallocHost((void**)&h_B, bytes));
    cudaCheckError(cudaMallocHost((void**)&h_C, bytes));
    cudaCheckError(cudaMallocHost((void**)&h_D, bytes));

    for (int i = 0; i < N * N; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        h_C[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C, *d_Temp, *d_D;
    cudaCheckError(cudaMalloc((void**)&d_A, bytes));
    cudaCheckError(cudaMalloc((void**)&d_B, bytes));
    cudaCheckError(cudaMalloc((void**)&d_C, bytes));
    cudaCheckError(cudaMalloc((void**)&d_Temp, bytes));
    cudaCheckError(cudaMalloc((void**)&d_D, bytes));

    cudaCheckError(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice));

    // 2. Execution Config
    // Block dim is reduced because each thread does more work
    // Threads per block: 16x16 = 256
    // But each block covers 64x64 output
    dim3 threadsPerBlock(16, 16); 
    dim3 blocksPerGrid((N + BM - 1) / BM, (N + BN - 1) / BN);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    std::cout << "Launching Register Tiled Kernels..." << std::endl;
    cudaEventRecord(start);

    // Kernel 1: Temp = A * B
    matrixMulRegisterTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_Temp, N);
    
    // Kernel 2: D = Temp * C
    matrixMulRegisterTiled<<<blocksPerGrid, threadsPerBlock>>>(d_Temp, d_C, d_D, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time: " << milliseconds << " ms" << std::endl;
    
    // Calculate TFLOPS (2 operations * N^3 * 2 calls)
    double flops = 2.0 * N * N * N * 2.0; 
    double gflops = (flops * 1e-9) / (milliseconds / 1000.0);
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    // Check errors and cleanup
    cudaCheckError(cudaGetLastError());
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_Temp); cudaFree(d_D);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C); cudaFreeHost(h_D);

    return 0;
}

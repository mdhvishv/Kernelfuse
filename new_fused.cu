#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

#define N 4096

// --- TUNING PARAMETERS (Optimized for OCCUPANCY) ---
#define BM 64
#define BN 64

// REDUCED BK TO 16
// Previously 32 (16KB buffers). Now 16 (8KB buffers).
// Total Shared Mem drops from 32KB -> 24KB.
// This allows more blocks to fit on the SM.
#define BK 16  

#define TM 4
#define TN 4

#define THREADS_X (BN/TN) 
#define THREADS_Y (BM/TM)

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// __launch_bounds__ helps the compiler optimize register usage
// (256 threads per block, min 2 blocks per SM)
__global__ void __launch_bounds__(256, 2) optimizedAtomicFusedGemm(
                                         const float4* __restrict__ A, 
                                         const float4* __restrict__ B, 
                                         const float4* __restrict__ C, 
                                         float* __restrict__ D) {
    // --------------------------------------------------------------------
    // SHARED MEMORY (Reduced Footprint)
    // --------------------------------------------------------------------
    // 16KB for AB Result
    __shared__ float s_AB[BM][BN]; 

    // 4KB + 4KB = 8KB for Loading Buffers (Previously 16KB)
    __shared__ float s_Load_A[BM][BK]; 
    __shared__ float s_Load_B[BK][BN];

    float r_load_A[TM];
    float r_load_B[TN];
    float r_acc[TM][TN]; 

    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int tid = ty * THREADS_X + tx; 

    // ====================================================================
    // PHASE 1: COMPUTE (A * B)
    // ====================================================================
    
    #pragma unroll
    for (int i = 0; i < TM; i++) 
        #pragma unroll
        for (int j = 0; j < TN; j++) r_acc[i][j] = 0.0f;

    for (int k = 0; k < N; k += BK) {
        
        // --- Vectorized Load (BK=16 -> 4 float4s) ---
        // With BK=16, we load 16 floats (64 bytes).
        // 256 threads loading 64x16 tile = 1024 floats = 256 float4s.
        // Each thread loads exactly 1 float4.
        
        int A_row_base = by * BM; 
        int A_col_base = k / 4; 

        // Load A (1 float4 per thread)
        // tile_idx range: 0..255. 
        // Tile A is 64x16 (floats) -> 64x4 (float4s) -> 256 elements.
        int row_A = tid / 4;   // 0..63
        int col_A = tid % 4;   // 0..3

        float4 vecA = A[(A_row_base + row_A) * (N/4) + (A_col_base + col_A)];
        s_Load_A[row_A][col_A*4 + 0] = vecA.x;
        s_Load_A[row_A][col_A*4 + 1] = vecA.y;
        s_Load_A[row_A][col_A*4 + 2] = vecA.z;
        s_Load_A[row_A][col_A*4 + 3] = vecA.w;

        // Load B (1 float4 per thread)
        // Tile B is 16x64 (floats) -> 16x16 (float4s) -> 256 elements.
        int B_row_base = k;
        int B_col_base = (bx * BN) / 4; 
        
        int row_B = tid / 16;  // 0..15
        int col_B = tid % 16;  // 0..15

        float4 vecB = B[(B_row_base + row_B) * (N/4) + (B_col_base + col_B)];
        s_Load_B[row_B][col_B*4 + 0] = vecB.x;
        s_Load_B[row_B][col_B*4 + 1] = vecB.y;
        s_Load_B[row_B][col_B*4 + 2] = vecB.z;
        s_Load_B[row_B][col_B*4 + 3] = vecB.w;

        __syncthreads();

        // --- Compute ---
        #pragma unroll
        for (int k_inner = 0; k_inner < BK; k_inner++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) r_load_A[i] = s_Load_A[ty * TM + i][k_inner];
            #pragma unroll
            for (int j = 0; j < TN; j++) r_load_B[j] = s_Load_B[k_inner][tx * TN + j];

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    r_acc[i][j] += r_load_A[i] * r_load_B[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            s_AB[ty * TM + i][tx * TN + j] = r_acc[i][j];
        }
    }
    __syncthreads();

    // ====================================================================
    // PHASE 2: MULTIPLY BY C
    // ====================================================================
    
    for (int c_blk = 0; c_blk < N; c_blk += BN) {
        
        #pragma unroll
        for (int i = 0; i < TM; i++) 
            #pragma unroll
            for (int j = 0; j < TN; j++) r_acc[i][j] = 0.0f;

        for (int k_step = 0; k_step < BN; k_step += BK) {
            
            // Load C (Reuse s_Load_B logic)
            // Tile size same as B: 16x64 (floats)
            int C_row_base = (bx * BN) + k_step;
            int C_col_base = c_blk / 4;

            int row_C = tid / 16; 
            int col_C = tid % 16; 

            float4 vecC = C[(C_row_base + row_C) * (N/4) + (C_col_base + col_C)];
            s_Load_B[row_C][col_C*4 + 0] = vecC.x;
            s_Load_B[row_C][col_C*4 + 1] = vecC.y;
            s_Load_B[row_C][col_C*4 + 2] = vecC.z;
            s_Load_B[row_C][col_C*4 + 3] = vecC.w;
            
            __syncthreads();

            // Compute
            #pragma unroll
            for (int k_inner = 0; k_inner < BK; k_inner++) {
                #pragma unroll
                for (int i = 0; i < TM; i++) {
                    r_load_A[i] = s_AB[ty * TM + i][k_step + k_inner];
                }
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    r_load_B[j] = s_Load_B[k_inner][tx * TN + j];
                }
                #pragma unroll
                for (int i = 0; i < TM; i++) {
                    #pragma unroll
                    for (int j = 0; j < TN; j++) {
                        r_acc[i][j] += r_load_A[i] * r_load_B[j];
                    }
                }
            }
            __syncthreads();
        }

        // Atomic Add
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                int global_D_Row = by * BM + (ty * TM + i);
                int global_D_Col = c_blk + (tx * TN + j);
                atomicAdd(&D[global_D_Row * N + global_D_Col], r_acc[i][j]);
            }
        }
    }
}

int main() {
    size_t bytes = N * N * sizeof(float);
    std::cout << "Running HIGH OCCUPANCY Kernel (BK=16, Reduced Shared Mem)..." << std::endl;
    
    float *h_A, *h_B, *h_C, *h_D;
    cudaMallocHost((void**)&h_A, bytes);
    cudaMallocHost((void**)&h_B, bytes);
    cudaMallocHost((void**)&h_C, bytes);
    cudaMallocHost((void**)&h_D, bytes);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand()%10)/10.0f; 
        h_B[i] = (float)(rand()%10)/10.0f; 
        h_C[i] = (float)(rand()%10)/10.0f; 
        h_D[i] = 0.0f;
    }

    float *d_A, *d_B, *d_C, *d_D;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes); cudaMalloc(&d_D, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_D, 0, bytes);

    dim3 threads(16, 16);
    dim3 blocks(N / 64, N / 64);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    optimizedAtomicFusedGemm<<<blocks, threads>>>(
        (float4*)d_A, (float4*)d_B, (float4*)d_C, d_D);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost);
    std::cout << "Result D[0]: " << h_D[0] << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C); cudaFreeHost(h_D);
    return 0;
}

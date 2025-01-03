#include <cblas.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#if NVIDIA
#include <cublas.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#if ROCM
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#endif

#if APPLE
#include <simd/simd.h>
#include <vecLib/vDSP.h>
#else
#include <immintrin.h>
#endif

#if NVIDIA
#define cuda_call(cuda_call_)                                                  \
  {                                                                            \
    auto cuda_res_ = cuda_call_;                                               \
    [[unlikely]] if (cuda_res_ != cudaSuccess) {                               \
      std::cout << "Failed in cuda call: " << #cuda_call_ << " : "             \
                << cuda_res_ << std::endl;                                     \
      std::abort();                                                            \
    }                                                                          \
  }
#endif
#if ROCM
#define hip_call(hip_call_)                                                    \
  {                                                                            \
    auto hip_res_ = hip_call_;                                                 \
    [[unlikely]] if (hip_res_ != hipSuccess) {                                 \
      std::cout << "Failed in hip call: " << #hip_call_ << " : " << hip_res_   \
                << std::endl;                                                  \
      std::abort();                                                            \
    }                                                                          \
  }
#define rocblas_call(rocblas_call_)                                            \
  {                                                                            \
    auto rocblas_res_ = rocblas_call_;                                         \
    [[unlikely]] if (rocblas_res_ != rocblas_status_success) {                 \
      std::cout << "Failed in call: " << #rocblas_call_ << std::endl;          \
      std::abort();                                                            \
    }                                                                          \
  }
#endif

constexpr size_t M = 2048;
constexpr size_t N = 2048;

void populate_values(float *a, float *b) {
  float iter = 1.0f;
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      a[i * M + j] = b[i * M + j] = iter++;
      if (iter > 10.0f) {
        iter = 1.0f;
      }
    }
  }
}

void transpose(float *res, float *in) {
  float iter = 1.0f;
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      res[i * M + j] = in[j * M + i];
    }
  }
}

void print_matrix(float *a) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      std::cout << a[i * M + j] << ", ";
    }
    std::cout << std::endl;
  }
}

void stupid_gemm(float *a, float *b, float *c) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      c[i * M + j] = 0.0f;
      for (size_t k = 0; k < M; k++) {
        c[i * M + j] += a[i * M + k] * b[k * M + j];
      }
    }
  }
}

void gemm_transpose(float *a, float *b, float *c) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      c[i * M + j] = 0.0f;
      for (size_t k = 0; k < M; k++) {
        c[i * M + j] += a[i * M + k] * b[j * M + k];
      }
    }
  }
}

void gemm_transpose_thread(float *a, float *b, float *c, size_t index,
                           size_t stride) {
  for (size_t i = index; i < index + stride; i++) {
    for (size_t j = 0; j < N; j++) {
      c[i * M + j] = 0.0f;
      for (size_t k = 0; k < M; k++) {
        c[i * M + j] += a[i * M + k] * b[j * M + k];
      }
    }
  }
}

void gemm_transpose_thread_simd_dot(float *a, float *b, float *c, size_t index,
                                    size_t stride) {
  for (size_t i = index; i < index + stride; i++) {
    for (size_t j = 0; j < N; j++) {
      c[i * M + j] = 0.0f;
      constexpr size_t fp_vec_size = 16;
      constexpr size_t buffer_size = M / fp_vec_size;
      float tmp_buffer[buffer_size];
#pragma unroll
      for (size_t k = 0, b_i = 0; k < M; k += fp_vec_size, b_i++) {
#if APPLE
        tmp_buffer[b_i] = simd::dot(*(simd::float16 *)(a + (i * M + k)),
                                    *(simd::float16 *)(b + (j * M + k)));
#endif
      }
#pragma unroll
      for (size_t x = 0; x < buffer_size; x += fp_vec_size) {
#if APPLE
        c[i * M + j] += simd::reduce_add(*(simd::float16 *)(tmp_buffer + x));
#endif
      }
    }
  }
}

bool validate(float *a, float *b) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      if (a[i * M + j] != b[i * M + j]) {
        std::cout << "mismatch: a[" << i * M << "][" << j
                  << "] : " << a[i * M + j] << " b[" << i * M << "][" << j
                  << "] : " << b[i * M + j] << std::endl;
        return false;
      }
    }
  }
  return true;
}

int main() {
  auto get_time = std::chrono::high_resolution_clock::now;
  float *a = new float[M * N];   // Input: a
  float *b = new float[M * N];   // Input: b
  float *t_b = new float[M * N]; // transposed b
  float *gemm_c = new float[M * N];
  float *my_c = new float[M * N];

  populate_values(a, b);
  transpose(t_b, b);

  // Run clbas first to get data we can match to
  auto blas_start = get_time();
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, M, 1.0f, a, M, b,
              M, 1.0f, gemm_c, M);
  auto blas_end = get_time();

#if NVIDIA
  float *d_a, *d_b, *d_c;
  cuda_call(cudaMalloc(&d_a, sizeof(float) * M * M));
  cuda_call(cudaMalloc(&d_b, sizeof(float) * M * M));
  cuda_call(cudaMalloc(&d_c, sizeof(float) * M * M));
  cuda_call(cudaMemcpy(d_a, a, sizeof(float) * M * N, cudaMemcpyHostToDevice));
  cuda_call(cudaMemcpy(d_b, b, sizeof(float) * M * N, cudaMemcpyHostToDevice));
  auto cublas_start = get_time();
  cublasSgemm('n', 'n', M, N, M, 1.0f, d_a, M, d_b, M, 1.0f, d_c, M);
  cuda_call(cudaDeviceSynchronize());
  auto cublas_end = get_time();
  cuda_call(
      cudaMemcpy(my_c, d_c, sizeof(float) * M * M, cudaMemcpyDeviceToHost));
  cuda_call(cudaFree(d_a));
  cuda_call(cudaFree(d_b));
  cuda_call(cudaFree(d_c));

  if (!validate(gemm_c, my_c)) {
    std::cout << "cublas failed" << std::endl;
    std::abort();
  }
#endif

#if ROCM
  rocblas_handle handle;
  rocblas_call(rocblas_create_handle(&handle));
  rocblas_operation transa = rocblas_operation_none,
                    transb = rocblas_operation_none;
  float *d_a, *d_b, *d_c;
  float alpha = 1.0f;
  hip_call(hipMalloc(&d_a, sizeof(float) * M * M));
  hip_call(hipMalloc(&d_b, sizeof(float) * M * M));
  hip_call(hipMalloc(&d_c, sizeof(float) * M * M));
  hip_call(hipMemcpy(d_a, a, sizeof(float) * M * N, hipMemcpyHostToDevice));
  hip_call(hipMemcpy(d_b, b, sizeof(float) * M * N, hipMemcpyHostToDevice));
  hip_call(hipDeviceSynchronize());
  auto rocblas_start = get_time();
  rocblas_call(rocblas_sgemm(handle, transa, transb, M, M, M, &alpha, d_a, M,
                             d_b, M, &alpha, d_c, M));
  auto rocblas_end = get_time();
  hip_call(hipMemcpy(my_c, d_c, sizeof(float) * M * M, hipMemcpyDeviceToHost));
  hip_call(hipFree(d_a));
  hip_call(hipFree(d_b));
  hip_call(hipFree(d_c));

  if (!validate(gemm_c, my_c)) {
    std::cout << "hipblas failed" << std::endl;
    std::abort();
  }
#endif

  // the most trivial gemm
  auto stupid_gemm_start = get_time();
  stupid_gemm(a, b, my_c);
  auto stupid_gemm_end = get_time();

  if (!validate(gemm_c, my_c)) {
    std::cout << "Trivial gemm failed" << std::endl;
    std::abort();
  }

  // clear result
  std::memset(my_c, 0, sizeof(float) * M * N);

  // GEMM but with transpose
  auto transpose_gemm_start = get_time();
  gemm_transpose(a, t_b, my_c);
  auto transpose_gemm_end = get_time();

  if (!validate(gemm_c, my_c)) {
    std::cout << "Transpose gemm failed" << std::endl;
    std::abort();
  }

  // clear result
  std::memset(my_c, 0, sizeof(float) * M * N);

  // GEMM transposed but on threaded
  auto thread_count = std::thread::hardware_concurrency() < 8 ? 4 : 8;
  auto transpose_gemm_thread_start = get_time();
  std::vector<std::thread> tgemm_threads;
  tgemm_threads.reserve(thread_count);
  for (size_t i = 0; i < thread_count; i++) {
    auto stride = (M / thread_count);
    tgemm_threads.emplace_back(
        std::thread(gemm_transpose_thread, a, t_b, my_c, i * stride, stride));
  }
  for (auto &thread : tgemm_threads) {
    thread.join();
  }
  auto transpose_gemm_thread_end = get_time();

  if (!validate(gemm_c, my_c)) {
    std::cout << "Transpose gemm thread failed" << std::endl;
    std::abort();
  }

  // clear result
  std::memset(my_c, 0, sizeof(float) * M * N);

#if APPLE
  // clear result
  std::memset(my_c, 0, sizeof(float) * M * N);

  // DSP
  auto dsp_start = get_time();
  vDSP_mmul(a, 1, b, 1, my_c, 1, M, N, N);
  auto dsp_end = get_time();

  if (!validate(gemm_c, my_c)) {
    std::cout << "Transpose gemm simd thread failed" << std::endl;
    std::abort();
  }
#endif

  auto blas_time =
      duration_cast<std::chrono::milliseconds>(blas_end - blas_start);
  auto stupid_gemm_time = duration_cast<std::chrono::milliseconds>(
      stupid_gemm_end - stupid_gemm_start);
  auto transpose_gemm_time = duration_cast<std::chrono::milliseconds>(
      transpose_gemm_end - transpose_gemm_start);
  auto transpose_gemm_thread_time = duration_cast<std::chrono::milliseconds>(
      transpose_gemm_thread_end - transpose_gemm_thread_start);
#if NVIDIA
  auto cublas_time =
      duration_cast<std::chrono::milliseconds>(cublas_end - cublas_start);
#endif
#if ROCM
  auto rocblas_time =
      duration_cast<std::chrono::milliseconds>(rocblas_end - rocblas_start);
#endif
#if APPLE
  auto dsp_time = duration_cast<std::chrono::milliseconds>(dsp_end - dsp_start);
#endif

  std::cout << "Matrix multiplication size: " << M << " * " << N << std::endl;
  std::cout << "Trivial   Gemm time                          : "
            << stupid_gemm_time << std::endl;
  std::cout << "Transpose matrix B Gemm time                 : "
            << transpose_gemm_time << std::endl;
  std::cout << "Transpose matrix B + threads Gemm time       : "
            << transpose_gemm_thread_time << std::endl;
  std::cout << "OpenBlas library gemm time                   : " << blas_time
            << std::endl;
#if NVIDIA
  std::cout << "cuBlas library gemm time                     : " << cublas_time
            << std::endl;
#endif
#if ROCM
  std::cout << "rocBlas library gemm time                    : " << rocblas_time
            << std::endl;
#endif
#if APPLE
  std::cout << "vDSP (Apple's DSP) time                      : " << dsp_time
            << std::endl;
#endif

  delete[] a;
  delete[] b;
  delete[] t_b;
  delete[] gemm_c;
  delete[] my_c;
}

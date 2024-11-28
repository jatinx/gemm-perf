#include <cblas.h>
#include <chrono>
#include <iostream>
#include <thread>

#if APPLE
#include <simd/simd.h>
#include <vecLib/vDSP.h>
#else
#include <immintrin.h>
#endif

constexpr size_t M = 1024;
constexpr size_t N = 1024;

void populate_values(float *a, float *b) {
  float iter = 1.0f;
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      a[i * M + j] = b[i * M + j] = iter++;
      if (iter > 100.0f) {
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

  // GEMM with simd and thread
  tgemm_threads.clear();
  auto transpose_gemm_thread_simd_dot_start = get_time();
  tgemm_threads.reserve(thread_count);
  for (size_t i = 0; i < thread_count; i++) {
    auto stride = (M / thread_count);
    tgemm_threads.emplace_back(std::thread(gemm_transpose_thread_simd_dot, a,
                                           t_b, my_c, i * stride, stride));
  }
  for (auto &thread : tgemm_threads) {
    thread.join();
  }
  auto transpose_gemm_thread_simd_dot_end = get_time();

  if (!validate(gemm_c, my_c)) {
    std::cout << "Transpose gemm simd thread failed" << std::endl;
    std::abort();
  }

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
  auto transpose_gemm_simd_thread_dot_time =
      duration_cast<std::chrono::milliseconds>(
          transpose_gemm_thread_simd_dot_end -
          transpose_gemm_thread_simd_dot_start);
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
  std::cout << "Transpose matrix B + threads + simd dot time : "
            << transpose_gemm_simd_thread_dot_time << std::endl;
  std::cout << "OpenBlas library gemm time                   : " << blas_time
            << std::endl;
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

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include "/usr/local/cuda-11.1/targets/x86_64-linux/include/cuda_fp16.h"

#define NNZ_EPSILON 1e-10

const std::string sparse_ver_str = "sp_v2";

template<typename T>
int GetNnzPadding(T* denseMatrix, int m, int n, int VECT = 4) {

  // then use loaded matrix count nnz with padding to prepare the buffer with padding csr convert.

  int nnz = 0;

  for (int i = 0; i < m; ++i) {
    const T *matrixRow = denseMatrix + i * n;
    size_t last_nnz = nnz;
    size_t save_last_idx = 0;

    for (int j = 0; j < n; ++j) {
      if (std::fabs(static_cast<float>(matrixRow[j])) > NNZ_EPSILON) {
        save_last_idx = j;
        ++nnz;
      }
    }

    // this padding may generate more zero.
    if ((nnz - last_nnz) % VECT != 0) {
      size_t padding = VECT - ((nnz - last_nnz) % VECT);
      for (size_t i = 0; i < padding; ++i) {
        ++nnz;
      }
    }
  }

  std::cout << "True sparsity[with Padding]: " << (1.0 - 1.0 * nnz / (m * n))
            << std::endl;

  return nnz;
}

template<typename T>
void DenseToSparsePadding(const T *denseMatrix, int m, int n, T *sparseMatrix,
                          int *colIdx, int *rowOffset, int VECT = 4) {
  float threshold = NNZ_EPSILON;
  size_t nnz = 0;
  rowOffset[0] = 0;

  for (int i = 0; i < m; ++i) {
    const T *matrixRow = denseMatrix + i * n;
    size_t last_nnz = nnz;

    for (int j = 0; j < n; ++j) {
      if (std::fabs(static_cast<float>(matrixRow[j])) > threshold) {
        colIdx[nnz] = j;
        sparseMatrix[nnz] = matrixRow[j];
        ++nnz;
      }
    }

    // this padding may generate more zero.
    if ((nnz - last_nnz) % VECT != 0) {
      size_t padding = VECT - ((nnz - last_nnz) % VECT);
      int last_idx = colIdx[nnz - 1];
      for (size_t i = 0; i < padding; ++i) {
        colIdx[nnz] = last_idx;
        sparseMatrix[nnz] = 0.0f;
        ++nnz;
      }
    }

    rowOffset[i + 1] = nnz;
  }
}

template<typename T>
void WriteToCache(const std::string &weight_file_path, int nnz,
                  int n_rows,
                  std::vector<T>& cpu_values,
                  std::vector<int>& cpu_col_indices,
                  std::vector<int>& cpu_row_offsets) {

  std::string cache_file_path = weight_file_path + "." + sparse_ver_str + ".cache";
  std::ofstream cache_write_stream(cache_file_path, std::ios::out | std::ios::binary);
  if (cache_write_stream) {
    cache_write_stream.write((char*)&nnz, sizeof(nnz));
    cache_write_stream.write((char*)cpu_values.data(), sizeof(cpu_values[0]) * nnz);
    cache_write_stream.write((char*)cpu_col_indices.data(), sizeof(cpu_col_indices[0]) * nnz);
    cache_write_stream.write((char*)cpu_row_offsets.data(), sizeof(cpu_row_offsets[0]) * (n_rows + 1));
    cache_write_stream.close();
    std::cout << "sparse cache created for " << cache_file_path << std::endl;
  } else {
    throw std::invalid_argument("cache file not writable.");
  }
  cache_write_stream.close();
}

template<typename T>
void DenseToCache(T* denseMatrix, int m, int n, std::string& weight_file_path) {
  int nnz = GetNnzPadding(denseMatrix, m, n);
  std::vector<T> cpu_values(nnz);
  std::vector<int> cpu_col_indices(nnz);
  std::vector<int> cpu_row_offsets(m + 1);
  DenseToSparsePadding(denseMatrix, m, n, cpu_values.data(), cpu_col_indices.data(), cpu_row_offsets.data());
  WriteToCache(weight_file_path, nnz, m, cpu_values, cpu_col_indices, cpu_row_offsets);
}

extern "C" void NumpyToCache(void* denseMatrix_p, int m, int n, char const* weight_file_path_bytes) {
  auto denseMatrix = reinterpret_cast<half*>(denseMatrix_p);
  std::string weight_file_path(weight_file_path_bytes);
  if (weight_file_path.find("cross_attention.key_value") != weight_file_path.npos) {
    std::string suffix_key = ".key";
    std::string suffix_value = ".value";
    weight_file_path += suffix_key;
    DenseToCache(denseMatrix, m / 2, n, weight_file_path);
    weight_file_path.erase(weight_file_path.size() - suffix_key.size());
    weight_file_path += suffix_value;
    DenseToCache(denseMatrix + m / 2 * n, m / 2, n, weight_file_path);
    // weight_file_path.erase(weight_file_path.size() - suffix_value.size());
  } else {
    DenseToCache(denseMatrix, m, n, weight_file_path);
  }
}

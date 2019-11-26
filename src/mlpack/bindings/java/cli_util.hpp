#ifndef MLPACK_BINDINGS_JAVA_CLI_UTIL_HPP
#define MLPACK_BINDINGS_JAVA_CLI_UTIL_HPP

#include <mlpack/core/util/cli.hpp>

namespace mlpack {
namespace util {

template <typename T>
void SetMatParam(const char* name, T* data, size_t rows, size_t columns)
{
  arma::Mat<T> m(data, rows, columns, false, true);
  CLI::GetParam<arma::Mat<T>>(name) = std::move(m);
}

template <typename T>
void SetRowParam(const char* name, T* data, size_t length)
{
  arma::Row<T> m(data, length, false, true);
  CLI::GetParam<arma::Row<T>>(name) = std::move(m);
}

template <typename T>
void SetColParam(const char* name, T* data, size_t length)
{
  arma::Col<T> m(data, length, false, true);
  CLI::GetParam<arma::Col<T>>(name) = std::move(m);
}

template <typename T>
void SetParam(const char* name, T value) 
{
  CLI::GetParam<T>(name) = std::move(value);
}

void SetMatWithInfoParam(const char* name, double* data, bool* info, size_t rows, size_t columns, bool pointsAreRows);

template <typename T>
T GetParam(const char* name)
{
  return std::move(CLI::GetParam<T>(name));
}

template <typename T>
typename T::elem_type* GetArmaParamData(const char* name)
{
  auto& param = CLI::GetParam<T>(name);
  
  if (param.mem && param.n_elem <= arma::arma_config::mat_prealloc) 
  {
    using R = typename T::elem_type;
    R* result = new R[param.n_elem];
    arma::arrayops::copy(result, param.mem, param.n_elem);
    return result;
  }

  arma::access::rw(param.mem_state) = 1;
  return param.memptr();
}

template <typename T>
size_t GetArmaParamRows(const char* name)
{
  return CLI::GetParam<T>(name).n_rows;
}

template <typename T>
size_t GetArmaParamColumns(const char* name)
{
  return CLI::GetParam<T>(name).n_cols;
}

template <typename T>
size_t GetArmaParamLength(const char* name)
{
  return CLI::GetParam<T>(name).n_elem;
}

template <typename T>
void SetVecElement(const char* name, int i, T element)
{
  CLI::GetParam<std::vector<T>>(name)[i] = std::move(element);
}

template <typename T>
void SetVecSize(const char* name, int size)
{
  CLI::GetParam<std::vector<T>>(name).resize(size);
}

template <typename T>
int GetVecSize(const char* name)
{
  return static_cast<int>(CLI::GetParam<std::vector<T>>(name).size());
}

template <typename T>
T GetVecElement(const char* name, int i)
{
  return std::move(CLI::GetParam<std::vector<T>>(name)[i]);
}

void SetPassed(const char* name);

void RestoreSettings(const char* name);

double* GetMatWithInfoParamData(const char* name);

size_t GetMatWithInfoParamCols(const char* name);

size_t GetMatWithInfoParamRows(const char* name);

size_t GetMatWithInfoParamLength(const char* name);

bool* GetMatWithInfoParamInfo(const char* name);

}
}

#endif

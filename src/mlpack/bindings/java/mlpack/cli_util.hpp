#ifndef MLPACK_BINDINGS_JAVA_MLPACK_CLI_UTIL_HPP
#define MLPACK_BINDINGS_JAVA_MLPACK_CLI_UTIL_HPP

#include <mlpack/core/util/cli.hpp>

namespace mlpack {
namespace util {

void DeleteArray(const void* p) 
{
  delete[] static_cast<const char*>(p);
}

template <typename T>
void SetMatParam(const char* name, const T* data, size_t rows, size_t columns)
{
  arma::Mat<T> m(data, rows, columns, false, true);
  CLI::GetParam<decltype(m)>(name) = std::move(m);
}

template <typename T>
void SetParam(const char* name, T value) 
{
  CLI::GetParam<T>(name) = std::move(value);
}

double* GetMatParamData(const char* name) 
{
  auto& param = CLI::GetParam<arma::mat>(name);
  
  if (param.mem && param.n_elem <= arma::arma_config::mat_prealloc) 
  {
    double* result = new double[param.n_elem];
    arma::arrayops::copy(result, param.mem, param.n_elem);
    return result;
  }

  arma::access::rw(param.mem_state) = 1;
  return param.memptr();
}

size_t GetMatParamRows(const char* name) 
{
  return CLI::GetParam<arma::mat>(name).n_rows;
}

size_t GetMatParamColumns(const char* name) 
{
  return CLI::GetParam<arma::mat>(name).n_cols;
}

size_t GetMatParamLength(const char* name) 
{
  return CLI::GetParam<arma::mat>(name).n_elem;
}

}
}

#endif
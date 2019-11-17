#ifndef MLPACK_BINDINGS_JAVA_CLI_UTIL_HPP
#define MLPACK_BINDINGS_JAVA_CLI_UTIL_HPP

#include <mlpack/core/util/cli.hpp>

namespace mlpack {
namespace util {

void DeleteArray(void* p);

template <typename T>
void SetMatParam(const char* name, T* data, size_t rows, size_t columns)
{
  arma::Mat<T> m(data, rows, columns, false, true);
  CLI::GetParam<arma::Mat<T>>(name) = std::move(m);
  CLI::SetPassed(name);
}

template <typename T>
void SetParam(const char* name, T value) 
{
  CLI::GetParam<T>(name) = std::move(value);
  CLI::SetPassed(name);
}

template <typename T>
T GetParam(const char* name)
{
  return std::move(CLI::GetParam<T>(name));
}

template <typename T>
T* GetMatParamData(const char* name)
{
  auto& param = CLI::GetParam<arma::Mat<T>>(name);
  
  if (param.mem && param.n_elem <= arma::arma_config::mat_prealloc) 
  {
    T* result = new T[param.n_elem];
    arma::arrayops::copy(result, param.mem, param.n_elem);
    return result;
  }

  arma::access::rw(param.mem_state) = 1;
  return param.memptr();
}

size_t GetMatParamRows(const char* name);

size_t GetMatParamColumns(const char* name);

size_t GetMatParamLength(const char* name);

void RestoreSettings(const char* name);

}
}

#endif

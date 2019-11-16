#ifndef MLPACK_BINDINGS_JAVA_CLI_UTIL_HPP
#define MLPACK_BINDINGS_JAVA_CLI_UTIL_HPP

#include <mlpack/core/util/cli.hpp>

namespace mlpack {
namespace util {

void DeleteArray(void* p);

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

double* GetMatParamData(const char* name);

size_t GetMatParamRows(const char* name);

size_t GetMatParamColumns(const char* name);

size_t GetMatParamLength(const char* name);

}
}

#endif

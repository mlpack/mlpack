#ifndef MLPACK_BINDINGS_JAVA_CLI_UTIL_HPP
#define MLPACK_BINDINGS_JAVA_CLI_UTIL_HPP

#include <mlpack/core/util/cli.hpp>

namespace mlpack {
namespace util {

template <typename T>
typename std::enable_if<!std::is_array<T>::value>::type Delete(void* p)
{
  delete static_cast<T*>(p);
}

template <typename T>
typename std::enable_if<std::is_array<T>::value>::type Delete(void* p)
{
  using U = typename std::remove_all_extents<T>::type;
  delete[] static_cast<U*>(p);
}

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

template <typename T>
T GetParam(const char* name)
{
  return std::move(CLI::GetParam<T>(name));
}

template <typename T>
auto GetArmaParamData(const char* name)
    -> decltype(CLI::GetParam<T>(name).memptr())
{
  using R = boost::remove_const_t<boost::remove_pointer_t<decltype(CLI::GetParam<T>(name).memptr())>>;
  auto& param = CLI::GetParam<T>(name);
  
  if (param.mem && param.n_elem <= arma::arma_config::mat_prealloc) 
  {
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

}
}

#endif

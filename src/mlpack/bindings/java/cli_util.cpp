#include "cli_util.hpp"

namespace mlpack {
namespace util {

void DeleteArray(void* p) 
{
  delete[] static_cast<unsigned char*>(p);
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

void RestoreSettings(const char* name)
{
  CLI::RestoreSettings(name);
}

}
}

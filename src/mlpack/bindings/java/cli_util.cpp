#include "cli_util.hpp"

namespace mlpack {
namespace util {

void DeleteArray(void* p) 
{
  delete[] static_cast<unsigned char*>(p);
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

#include "cli_util.hpp"

namespace mlpack {
namespace util {

void SetPassed(const char* name)
{
  CLI::SetPassed(name);
}

void RestoreSettings(const char* name)
{
  CLI::RestoreSettings(name);
}

bool* GetMatWithInfoParamInfo(const char* name)
{
  const data::DatasetInfo& info = std::get<0>(
      CLI::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(name));

  size_t n = info.Dimensionality();
  bool* result = new bool[n];

  for (size_t i = 0; i < n; ++i)
  {
    result[i] = info.Type(i) == data::Datatype::categorical;
  }

  return result;
}

size_t GetMatWithInfoParamLength(const char* name)
{
  return std::get<1>(
      CLI::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(name)).n_elem;
}

double* GetMatWithInfoParamData(const char* name)
{
  arma::mat& param = std::get<1>(
      CLI::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(name));

  if (param.mem && param.n_elem <= arma::arma_config::mat_prealloc)
  {
    double* result = new double[param.n_elem];
    arma::arrayops::copy(result, param.mem, param.n_elem);
    return result;
  }

  arma::access::rw(param.mem_state) = 1;
  return param.memptr();
}

size_t GetMatWithInfoParamCols(const char* name)
{
  return std::get<1>(
      CLI::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(name)).n_cols;
}

size_t GetMatWithInfoParamRows(const char* name)
{
  return std::get<1>(
      CLI::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(name)).n_rows;
}

void SetMatWithInfoParam(
    const char* name, double* data, bool* info, size_t rows, size_t columns)
{
  data::DatasetInfo d(rows);
  for (size_t i = 0; i < d.Dimensionality(); ++i)
  {
    d.Type(i) = (info[i]) ? data::Datatype::categorical :
        data::Datatype::numeric;
  }

  arma::mat m(data, rows, columns, false, true);
  std::get<0>(CLI::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
      name)) = std::move(d);
  std::get<1>(CLI::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
      name)) = std::move(m);
}

void EnableVerbose()
{
  Log::Info.ignoreInput = false;
}

void DisableVerbose()
{
  Log::Info.ignoreInput = true;
}

} // namespace util
} // namespace mlpack

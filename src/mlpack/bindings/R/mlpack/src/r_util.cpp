/**
 * @file src/r_util.cpp
 * @author Yashwant Singh Parihar
 *
 * Utility functions for R-bindings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <rcpp_mlpack.h>
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>

using namespace mlpack;
using namespace Rcpp;

template<typename eT>
bool inline inplace_transpose(arma::Mat<eT>& X)
{
  try
  {
    X = arma::trans(X);
    return false;
  }
  catch (std::bad_alloc&)
  {
    return true;
  }
}

// Call IO::RestoreSettings() for a given program name.
// [[Rcpp::export]]
void IO_RestoreSettings(const std::string& programName)
{
  IO::RestoreSettings(programName);
}

// Call IO::SetParam<int>().
// [[Rcpp::export]]
void IO_SetParamInt(const std::string& paramName, int paramValue)
{
  IO::GetParam<int>(paramName) = paramValue;
  IO::SetPassed(paramName);
}

// Call IO::SetParam<double>().
// [[Rcpp::export]]
void IO_SetParamDouble(const std::string& paramName, double paramValue)
{
  IO::GetParam<double>(paramName) = paramValue;
  IO::SetPassed(paramName);
}

// Call IO::SetParam<std::string>().
// [[Rcpp::export]]
void IO_SetParamString(const std::string& paramName, std::string& paramValue)
{
  IO::GetParam<std::string>(paramName) = paramValue;
  IO::SetPassed(paramName);
}

// Call IO::SetParam<bool>().
// [[Rcpp::export]]
void IO_SetParamBool(const std::string& paramName, bool paramValue)
{
  IO::GetParam<bool>(paramName) = paramValue;
  IO::SetPassed(paramName);
}

// Call IO::SetParam<std::vector<std::string>>().
// [[Rcpp::export]]
void IO_SetParamVecString(const std::string& paramName,
                          const std::vector<std::string>& str)
{
  IO::GetParam<std::vector<std::string>>(paramName) = std::move(str);
  IO::SetPassed(paramName);
}

// Call IO::SetParam<std::vector<int>>().
// [[Rcpp::export]]
void IO_SetParamVecInt(const std::string& paramName,
                       const std::vector<int>& ints)
{
  IO::GetParam<std::vector<int>>(paramName) = std::move(ints);
  IO::SetPassed(paramName);
}

// Call IO::SetParam<arma::mat>().
// [[Rcpp::export]]
void IO_SetParamMat(const std::string& paramName,
                    const arma::mat& paramValue)
{
  IO::GetParam<arma::mat>(paramName) = paramValue.t();
  IO::SetPassed(paramName);
}

// Call IO::SetParam<arma::Mat<size_t>>().
// [[Rcpp::export]]
void IO_SetParamUMat(const std::string& paramName,
                     const arma::Mat<size_t>& paramValue)
{
  IO::GetParam<arma::Mat<size_t>>(paramName) = paramValue.t();
  IO::SetPassed(paramName);
}

// Call IO::SetParam<arma::rowvec>().
// [[Rcpp::export]]
void IO_SetParamRow(const std::string& paramName,
                    const arma::rowvec& paramValue)
{
  IO::GetParam<arma::rowvec>(paramName) = std::move(paramValue);
  IO::SetPassed(paramName);
}

// Call IO::SetParam<arma::Row<size_t>>().
// [[Rcpp::export]]
void IO_SetParamURow(const std::string& paramName,
                     const arma::Row<size_t>& paramValue)
{
  IO::GetParam<arma::Row<size_t>>(paramName) = paramValue - 1;
  IO::SetPassed(paramName);
}

// Call IO::SetParam<arma::vec>().
// [[Rcpp::export]]
void IO_SetParamCol(const std::string& paramName,
                    const arma::vec& paramValue)
{
  IO::GetParam<arma::vec>(paramName) = std::move(paramValue);
  IO::SetPassed(paramName);
}

// Call IO::SetParam<arma::Col<size_t>>().
// [[Rcpp::export]]
void IO_SetParamUCol(const std::string& paramName,
                     const arma::Col<size_t>& paramValue)
{
  IO::GetParam<arma::Col<size_t>>(paramName) = paramValue - 1;
  IO::SetPassed(paramName);
}

// Call IO::SetParam<std::tuple<data::DatasetInfo, arma::mat>>().
// [[Rcpp::export]]
void IO_SetParamMatWithInfo(const std::string& paramName,
                            const LogicalVector& dimensions,
                            const arma::mat& paramValue)
{
  data::DatasetInfo d(paramValue.n_cols);
  for (size_t i = 0; i < d.Dimensionality(); ++i)
  {
    d.Type(i) = (dimensions[i]) ? data::Datatype::categorical :
        data::Datatype::numeric;
  }
  std::get<0>(IO::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
      paramName)) = std::move(d);
  std::get<1>(IO::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(
      paramName)) = paramValue.t();
  IO::SetPassed(paramName);
}

// Call IO::GetParam<int>().
// [[Rcpp::export]]
int IO_GetParamInt(const std::string& paramName)
{
  return IO::GetParam<int>(paramName);
}

// Call IO::GetParam<double>().
// [[Rcpp::export]]
double IO_GetParamDouble(const std::string& paramName)
{
  return IO::GetParam<double>(paramName);
}

// Call IO::GetParam<std::string>().
// [[Rcpp::export]]
std::string& IO_GetParamString(const std::string& paramName)
{
  return IO::GetParam<std::string>(paramName);
}

// Call IO::GetParam<bool>().
// [[Rcpp::export]]
bool IO_GetParamBool(const std::string& paramName)
{
  return IO::GetParam<bool>(paramName);
}

// Call IO::GetParam<std::vector<std::string>>().
// [[Rcpp::export]]
const std::vector<std::string>& IO_GetParamVecString(const
                                    std::string& paramName)
{
  return std::move(IO::GetParam<std::vector<std::string>>(paramName));
}

// Call IO::GetParam<std::vector<int>>().
// [[Rcpp::export]]
const std::vector<int>& IO_GetParamVecInt(const std::string& paramName)
{
  return std::move(IO::GetParam<std::vector<int>>(paramName));
}

// Call IO::GetParam<arma::mat>().
// [[Rcpp::export]]
const arma::mat& IO_GetParamMat(const std::string& paramName)
{
  inplace_transpose(IO::GetParam<arma::mat>(paramName));
  return std::move(IO::GetParam<arma::mat>(paramName));
}

// Call IO::GetParam<arma::Mat<size_t>>().
// [[Rcpp::export]]
const arma::Mat<size_t>& IO_GetParamUMat(const std::string& paramName)
{
  inplace_transpose(IO::GetParam<arma::Mat<size_t>>(paramName));
  return std::move(IO::GetParam<arma::Mat<size_t>>(paramName));
}

// Call IO::GetParam<arma::rowvec>().
// [[Rcpp::export]]
const arma::vec IO_GetParamRow(const std::string& paramName)
{
  return IO::GetParam<arma::rowvec>(paramName).t();
}

// Call IO::GetParam<arma::Row<size_t>>().
// [[Rcpp::export]]
const arma::Col<size_t> IO_GetParamURow(const std::string& paramName)
{
  return IO::GetParam<arma::Row<size_t>>(paramName).t() + 1;
}

// Call IO::GetParam<arma::vec>().
// [[Rcpp::export]]
const arma::rowvec IO_GetParamCol(const std::string& paramName)
{
  return IO::GetParam<arma::vec>(paramName).t();
}

// Call IO::GetParam<arma::Col<size_t>>().
// [[Rcpp::export]]
const arma::Row<size_t> IO_GetParamUCol(const std::string& paramName)
{
  return IO::GetParam<arma::Col<size_t>>(paramName).t() + 1;
}

// Call IO::GetParam<std::tuple<data::DatasetInfo, arma::mat>>().
// [[Rcpp::export]]
List IO_GetParamMatWithInfo(const std::string& paramName)
{
  const data::DatasetInfo& d = std::get<0>(
      IO::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(paramName));
  const arma::mat& m = std::get<1>(
      IO::GetParam<std::tuple<data::DatasetInfo, arma::mat>>(paramName)).t();

  LogicalVector dims(d.Dimensionality());
  for (size_t i = 0; i < d.Dimensionality(); ++i)
    dims[i] = (d.Type(i) == data::Datatype::numeric) ? false : true;

  return List::create (Rcpp::Named("Info") = std::move(dims),
                       Rcpp::Named("Data") = std::move(m));
}

// Enable verbose output.
// [[Rcpp::export]]
void IO_EnableVerbose()
{
  Log::Info.ignoreInput = false;
}

// Disable verbose output.
// [[Rcpp::export]]
void IO_DisableVerbose()
{
  Log::Info.ignoreInput = true;
}

// Reset the state of all timers.
// [[Rcpp::export]]
void IO_ResetTimers()
{
  IO::GetSingleton().timer.Reset();
}

// Set an argument as passed to the IO object.
// [[Rcpp::export]]
void IO_SetPassed(const std::string& paramName)
{
  IO::SetPassed(paramName);
}

// Clear settings.
// [[Rcpp::export]]
void IO_ClearSettings()
{
  IO::ClearSettings();
}

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

// Create a new util::Params object.
// [[Rcpp::export]]
SEXP CreateParams(const std::string& bindingName)
{
  util::Params* p = new util::Params(IO::Parameters(bindingName));
  return std::move(Rcpp::XPtr<util::Params>(p));
}

// Create a new util::Timers object.
// [[Rcpp::export]]
SEXP CreateTimers()
{
  util::Timers* t = new util::Timers();
  return std::move(Rcpp::XPtr<util::Timers>(t));
}

// Call params.Get<int>() to set the value of a parameter.
// [[Rcpp::export]]
void SetParamInt(SEXP params, const std::string& paramName, int paramValue)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  p.Get<int>(paramName) = paramValue;
  p.SetPassed(paramName);
}

// Call params.Get<double>() to set the value of a parameter.
// [[Rcpp::export]]
void SetParamDouble(SEXP params,
                    const std::string& paramName,
                    double paramValue)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  p.Get<double>(paramName) = paramValue;
  p.SetPassed(paramName);
}

// Call params.Get<std::string>() to set the value of a parameter.
// [[Rcpp::export]]
void SetParamString(SEXP params,
                    const std::string& paramName,
                    std::string& paramValue)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  p.Get<std::string>(paramName) = paramValue;
  p.SetPassed(paramName);
}

// Call params.Get<bool>() to set the value of a parameter.
// [[Rcpp::export]]
void SetParamBool(SEXP params, const std::string& paramName, bool paramValue)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  p.Get<bool>(paramName) = paramValue;
  p.SetPassed(paramName);
}

// Call params.Get<std::vector<std::string>>() to set the value of a parameter.
// [[Rcpp::export]]
void SetParamVecString(SEXP params,
                       const std::string& paramName,
                       const std::vector<std::string>& str)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  p.Get<std::vector<std::string>>(paramName) = std::move(str);
  p.SetPassed(paramName);
}

// Call params.Get<std::vector<int>>() to set the value of a parameter.
// [[Rcpp::export]]
void SetParamVecInt(SEXP params,
                    const std::string& paramName,
                    const std::vector<int>& ints)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  p.Get<std::vector<int>>(paramName) = std::move(ints);
  p.SetPassed(paramName);
}

// Call params.Get<arma::mat>() to set the value of a parameter.
// [[Rcpp::export]]
void SetParamMat(SEXP params,
                 const std::string& paramName,
                 const arma::mat& paramValue,
                 bool transpose)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  p.Get<arma::mat>(paramName) = (transpose ? paramValue.t() : paramValue);
  p.SetPassed(paramName);
}

// Call params.Get<arma::Mat<size_t>>() to set the value of a parameter.
// [[Rcpp::export]]
void SetParamUMat(SEXP params,
                  const std::string& paramName,
                  const arma::Mat<size_t>& paramValue)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  p.Get<arma::Mat<size_t>>(paramName) = paramValue.t();
  p.SetPassed(paramName);
}

// Call params.Get<arma::rowvec>() to set the value of a parameter.
// [[Rcpp::export]]
void SetParamRow(SEXP params,
                 const std::string& paramName,
                 const arma::rowvec& paramValue)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  p.Get<arma::rowvec>(paramName) = std::move(paramValue);
  p.SetPassed(paramName);
}

// Call params.Get<arma::Row<size_t>>() to set the value of a parameter.
// [[Rcpp::export]]
void SetParamURow(SEXP params,
                  const std::string& paramName,
                  const arma::Row<size_t>& paramValue)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);

  // Check for zeros in the input---if we received these, the user is mistaken,
  // because in R labels should start from 1.
  if (arma::any(paramValue == 0))
  {
    Log::Fatal << "When passing labels from R to mlpack, labels should be in "
        << "the range from 1 to the number of classes!" << std::endl;
  }

  p.Get<arma::Row<size_t>>(paramName) = paramValue - 1;
  p.SetPassed(paramName);
}

// Call params.Get<arma::vec>() to set the value of a parameter.
// [[Rcpp::export]]
void SetParamCol(SEXP params,
                 const std::string& paramName,
                 const arma::vec& paramValue)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  p.Get<arma::vec>(paramName) = std::move(paramValue);
  p.SetPassed(paramName);
}

// Call params.Get<arma::Col<size_t>>() to set the value of a parameter.
// [[Rcpp::export]]
void SetParamUCol(SEXP params,
                  const std::string& paramName,
                  const arma::Col<size_t>& paramValue)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);

  // Check for zeros in the input---if we received these, the user is mistaken,
  // because in R labels should start from 1.
  if (arma::any(paramValue == 0))
  {
    Log::Fatal << "When passing labels from R to mlpack, labels should be in "
        << "the range from 1 to the number of classes!" << std::endl;
  }

  p.Get<arma::Col<size_t>>(paramName) = paramValue - 1;
  p.SetPassed(paramName);
}

// Call params.Get<std::tuple<data::DatasetInfo, arma::mat>>() to set the value
// of a parameter.
// [[Rcpp::export]]
void SetParamMatWithInfo(SEXP params,
                         const std::string& paramName,
                         const LogicalVector& dimensions,
                         const arma::mat& paramValue)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  data::DatasetInfo d(paramValue.n_cols);
  bool hasCategoricals = false;
  for (size_t i = 0; i < d.Dimensionality(); ++i)
  {
    d.Type(i) = (dimensions[i]) ? data::Datatype::categorical :
        data::Datatype::numeric;
    if (dimensions[i])
      hasCategoricals = true;
  }

  arma::mat m = paramValue.t();

  // Do we need to find how many categories we have?
  if (hasCategoricals)
  {
    arma::vec maxs = arma::max(paramValue, 1) + 1;

    for (size_t i = 0; i < d.Dimensionality(); ++i)
    {
      if (dimensions[i])
      {
        // Map the right number of objects.
        for (size_t j = 0; j < (size_t) maxs[i]; ++j)
        {
          std::ostringstream oss;
          oss << j;
          d.MapString<double>(oss.str(), i);
        }
      }
    }
  }

  std::get<0>(p.Get<std::tuple<data::DatasetInfo, arma::mat>>(
      paramName)) = std::move(d);
  std::get<1>(p.Get<std::tuple<data::DatasetInfo, arma::mat>>(
      paramName)) = std::move(m);
  p.SetPassed(paramName);
}

// Call p.Get<int>().
// [[Rcpp::export]]
int GetParamInt(SEXP params, const std::string& paramName)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  return p.Get<int>(paramName);
}

// Call p.Get<double>().
// [[Rcpp::export]]
double GetParamDouble(SEXP params, const std::string& paramName)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  return p.Get<double>(paramName);
}

// Call p.Get<std::string>().
// [[Rcpp::export]]
std::string& GetParamString(SEXP params, const std::string& paramName)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  return p.Get<std::string>(paramName);
}

// Call p.Get<bool>().
// [[Rcpp::export]]
bool GetParamBool(SEXP params, const std::string& paramName)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  return p.Get<bool>(paramName);
}

// Call p.Get<std::vector<std::string>>().
// [[Rcpp::export]]
const std::vector<std::string>& GetParamVecString(
    SEXP params,
    const std::string& paramName)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  return std::move(p.Get<std::vector<std::string>>(paramName));
}

// Call p.Get<std::vector<int>>().
// [[Rcpp::export]]
const std::vector<int>& GetParamVecInt(SEXP params,
                                       const std::string& paramName)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  return std::move(p.Get<std::vector<int>>(paramName));
}

// Call p.Get<arma::mat>().
// [[Rcpp::export]]
const arma::mat& GetParamMat(SEXP params, const std::string& paramName)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  inplace_transpose(p.Get<arma::mat>(paramName));
  return std::move(p.Get<arma::mat>(paramName));
}

// Call p.Get<arma::Mat<size_t>>().
// [[Rcpp::export]]
const arma::Mat<size_t>& GetParamUMat(SEXP params,
                                      const std::string& paramName)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  inplace_transpose(p.Get<arma::Mat<size_t>>(paramName));
  return std::move(p.Get<arma::Mat<size_t>>(paramName));
}

// Call p.Get<arma::rowvec>().
// [[Rcpp::export]]
const arma::vec GetParamRow(SEXP params, const std::string& paramName)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  return p.Get<arma::rowvec>(paramName).t();
}

// Call p.Get<arma::Row<size_t>>().
// [[Rcpp::export]]
const arma::Col<size_t> GetParamURow(SEXP params,
                                     const std::string& paramName)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  return p.Get<arma::Row<size_t>>(paramName).t() + 1;
}

// Call p.Get<arma::vec>().
// [[Rcpp::export]]
const arma::rowvec GetParamCol(SEXP params, const std::string& paramName)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  return p.Get<arma::vec>(paramName).t();
}

// Call p.Get<arma::Col<size_t>>().
// [[Rcpp::export]]
const arma::Row<size_t> GetParamUCol(SEXP params,
                                     const std::string& paramName)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  return p.Get<arma::Col<size_t>>(paramName).t() + 1;
}

// Call p.Get<std::tuple<data::DatasetInfo, arma::mat>>().
// [[Rcpp::export]]
List IO_GetParamMatWithInfo(SEXP params, const std::string& paramName)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  const data::DatasetInfo& d = std::get<0>(
      p.Get<std::tuple<data::DatasetInfo, arma::mat>>(paramName));
  const arma::mat& m = std::get<1>(
      p.Get<std::tuple<data::DatasetInfo, arma::mat>>(paramName)).t();

  LogicalVector dims(d.Dimensionality());
  for (size_t i = 0; i < d.Dimensionality(); ++i)
    dims[i] = (d.Type(i) == data::Datatype::numeric) ? false : true;

  return List::create(Rcpp::Named("Info") = std::move(dims),
                      Rcpp::Named("Data") = std::move(m));
}

// Enable verbose output.
// [[Rcpp::export]]
void EnableVerbose()
{
  Log::Info.ignoreInput = false;
}

// Disable verbose output.
// [[Rcpp::export]]
void DisableVerbose()
{
  Log::Info.ignoreInput = true;
}

// Reset the state of all timers.
// [[Rcpp::export]]
void ResetTimers()
{
  Timer::ResetAll();
}

// Set an argument as passed to the IO object.
// [[Rcpp::export]]
void SetPassed(SEXP params, const std::string& paramName)
{
  util::Params& p = *Rcpp::as<Rcpp::XPtr<util::Params>>(params);
  p.SetPassed(paramName);
}

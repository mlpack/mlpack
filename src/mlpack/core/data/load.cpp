/**
 * @file core/data/load.cpp
 * @author Tham Ngap Wei
 *
 * Force instantiation of some Load() overloads to reduce compile time.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "load.hpp"
#include "load_impl.hpp"

namespace mlpack {
namespace data /** Functions to load and save matrices and models. */ {

template bool Load<int>(const std::string&,
                        arma::Mat<int>&,
                        const bool,
                        const bool,
                        const arma::file_type);

template bool Load<unsigned int>(const std::string&,
                                 arma::Mat<unsigned int>&,
                                 const bool,
                                 const bool,
                                 const arma::file_type);

template bool Load<unsigned long>(const std::string&,
                                  arma::Mat<unsigned long>&,
                                  const bool,
                                  const bool,
                                  const arma::file_type);

template bool Load<unsigned long long>(const std::string&,
                                       arma::Mat<unsigned long long>&,
                                       const bool,
                                       const bool,
                                       const arma::file_type);

template bool Load<float>(const std::string&,
                          arma::Mat<float>&,
                          const bool,
                          const bool,
                          const arma::file_type);

template bool Load<double>(const std::string&,
                           arma::Mat<double>&,
                           const bool,
                           const bool,
                           const arma::file_type);

template bool Load<int>(const std::string&,
                        arma::SpMat<int>&,
                        const bool,
                        const bool);

template bool Load<unsigned int>(const std::string&,
                                 arma::SpMat<unsigned int>&,
                                 const bool,
                                 const bool);

template bool Load<unsigned long>(const std::string&,
                                  arma::SpMat<unsigned long>&,
                                  const bool,
                                  const bool);

template bool Load<unsigned long long>(const std::string&,
                                       arma::SpMat<unsigned long long>&,
                                       const bool,
                                       const bool);

template bool Load<double>(const std::string&,
                           arma::SpMat<double>&,
                           const bool,
                           const bool);

template bool Load<float>(const std::string&,
                          arma::SpMat<float>&,
                          const bool,
                          const bool);

template bool Load<int, IncrementPolicy>(const std::string&,
                                         arma::Mat<int>&,
                                         DatasetMapper<IncrementPolicy>&,
                                         const bool,
                                         const bool);

template bool Load<unsigned int, IncrementPolicy>(
    const std::string&,
    arma::Mat<unsigned int>&,
    DatasetMapper<IncrementPolicy>&,
    const bool,
    const bool);

template bool Load<unsigned long, IncrementPolicy>(
    const std::string&,
    arma::Mat<unsigned long>&,
    DatasetMapper<IncrementPolicy>&,
    const bool,
    const bool);

template bool Load<unsigned long long, IncrementPolicy>(
    const std::string&,
    arma::Mat<unsigned long long>&,
    DatasetMapper<IncrementPolicy>&,
    const bool,
    const bool);

template bool Load<float, IncrementPolicy>(const std::string&,
                                           arma::Mat<float>&,
                                           DatasetMapper<IncrementPolicy>&,
                                           const bool,
                                           const bool);

template bool Load<double, IncrementPolicy>(const std::string&,
                                            arma::Mat<double>&,
                                            DatasetMapper<IncrementPolicy>&,
                                            const bool,
                                            const bool);

} // namespace data
} // namespace mlpack

#include "load.hpp"
#include "load_impl.hpp"

namespace mlpack {
namespace data /** Functions to load and save matrices and models. */ {

template bool Load<int>(const std::string&, arma::Mat<int>&, const bool, const bool);
template bool Load<size_t>(const std::string&, arma::Mat<size_t>&, const bool, const bool);
template bool Load<float>(const std::string&, arma::Mat<float>&, const bool, const bool);
template bool Load<double>(const std::string&, arma::Mat<double>&, const bool, const bool);
template bool Load<unsigned long long>(const std::string&, arma::Mat<unsigned long long>&, const bool, const bool);

template bool Load<int, IncrementPolicy>(const std::string&, arma::Mat<int>&,
                                         DatasetMapper<IncrementPolicy>&,
                                         const bool, const bool);

template bool Load<size_t, IncrementPolicy>(const std::string&, arma::Mat<size_t>&,
                                            DatasetMapper<IncrementPolicy>&,
                                            const bool, const bool);

template bool Load<float, IncrementPolicy>(const std::string&, arma::Mat<float>&,
                                           DatasetMapper<IncrementPolicy>&,
                                           const bool, const bool);

template bool Load<double, IncrementPolicy>(const std::string&, arma::Mat<double>&,
                                            DatasetMapper<IncrementPolicy>&,
                                            const bool, const bool);

template bool Load<unsigned long long, IncrementPolicy>(const std::string&, arma::Mat<unsigned long long>&,
                                                        DatasetMapper<IncrementPolicy>&,
                                                        const bool, const bool);

}}

#ifndef MLPACK_CORE_DATA_NEW_PARSER_HPP
#define MLPACK_CORE_DATA_NEW_PARSER_HPP

namespace mlpack {
namespace data {

template<typename eT>
inline
bool
convert_token(eT& val, const std::string& token);

template<typename T>
inline
bool
convert_token(std::complex<T>& val, const std::string& token);

template<typename eT>
inline
bool
load_csv_ascii(arma::Mat<eT>& x, std::istream& f, std::string&);

template<typename eT>
inline
arma_cold
bool
load_data(const std::string name, const arma::file_type type);


    }   // namespace data
}   // namepsace mlpack

// Include implementation
#include "new_parser_impl.hpp"

#endif

/**
 * @file type_names.hpp
 * @author Ryan Curtin
 *
 * Given a primitive type, return the Python name for that type.  There's no
 * need for template metaprogramming here, because it doesn't really matter how
 * quickly we can generate the bindings.
 */
#ifndef MLPACK_BINDINGS_PYTHON_TYPE_NAMES_HPP
#define MLPACK_BINDINGS_PYTHON_TYPE_NAMES_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace bindings {
namespace python {

/**
 * Get the Python typename.
 */
inline std::string Typename(const util::ParamData& data)
{
  if (TYPENAME(int) == data.tname)
    return "int";
  else if (TYPENAME(float) == data.tname)
    return "float";
  else if (TYPENAME(double) == data.tname)
    return "double";
  else if (TYPENAME(std::string) == data.tname)
    return "string";
  else if (TYPENAME(bool) == data.tname)
    return "bool"; // This refers to the cimported libcpp.bool type from Cython.
  else if (TYPENAME(std::vector<int>) == data.tname)
    return "array"; // Right?
  else if (TYPENAME(std::vector<float>) == data.tname)
    return "array";
  else if (TYPENAME(std::vector<double>) == data.tname)
    return "array";
  else if (TYPENAME(std::vector<std::string>) == data.tname)
    return "array";
  else if (TYPENAME(std::vector<bool>) == data.tname)
    return "array";
  else if (TYPENAME(arma::mat) == data.tname)
    return "arma.Mat[double]";
  else if (TYPENAME(arma::Mat<size_t>) == data.tname)
    return "arma.Mat[size_t]";
//  else if (TYPENAME(std::tuple<data::DatasetInfo, arma::mat>) == data.tname)
//    return "tuple";
  else
    return "unknown"; // Let's hope this never happens.
}

/**
 * Determine if this is a matrix type.
 */
inline bool IsMatrixType(const util::ParamData& data)
{
  if ((TYPENAME(arma::mat) == data.tname) ||
      (TYPENAME(arma::Mat<size_t>) == data.tname))
    return true;
  else
    return false;
}

/**
 * Return the suffix for the numpy/Armadillo transition functions based on the
 * type of the matrix.
 */
inline char MatrixTypeSuffix(const util::ParamData& data)
{
  if (TYPENAME(arma::mat) == data.tname)
    return 'd';
  else if (TYPENAME(arma::Mat<size_t>) == data.tname)
    return 's';
  else
    throw std::invalid_argument("MatrixTypeSuffix(): unknown typename " +
        data.tname);
}

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif

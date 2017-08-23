/**
 * @file get_arma_type.hpp
 * @author Ryan Curtin
 *
 * Return "mat", "col", or "row" depending on the type of the given Armadillo
 * object.  This is so that the correct overload of arma_numpy.numpy_to_<type>()
 * can be called.
 */
#ifndef MLPACK_BINDINGS_PYTHON_GET_ARMA_TYPE_HPP
#define MLPACK_BINDINGS_PYTHON_GET_ARMA_TYPE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace python {

/**
 * This is used for arma::Mat<> types; it will return "mat" for matrices, "row"
 * for row vectors, and "col" for column vectors.
 */
template<typename T>
inline std::string GetArmaType()
{
  if (T::is_col)
    return "col";
  else if (T::is_row)
    return "row";
  else
    return "mat";
}

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif

/**
 * @file get_param.hpp
 * @author Ryan Curtin
 *
 * Get a parameter for a Python binding.
 */
#ifndef MLPACK_BINDINGS_PYTHON_GET_PARAM_HPP
#define MLPACK_BINDINGS_PYTHON_GET_PARAM_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace python {

/**
 * All Python binding types are exactly what is held in the ParamData, so no
 * special handling is necessary.
 */
template<typename T>
void GetParam(const util::ParamData& d,
              const void* /* input */,
              void* output)
{
//  typedef typename std::remove_pointer<T>::type TRaw;
//  if (std::is_pointer<T>::value) // If true, this is a model.
//  {
//    std::cout << "get a raw pointer for " << d.name << ": " << boost::any_cast<TRaw*>(d.value) <<
//"\n";
//    *((TRaw***) output) = const_cast<TRaw**>(boost::any_cast<TRaw**>(&d.value));
//  }
//  else
  {
    *((T**) output) = const_cast<T*>(boost::any_cast<T>(&d.value));
  }
}

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif

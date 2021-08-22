/**
 * @file bindings/python/print_class_defn.hpp
 * @author Ryan Curtin
 *
 * Print the class definition for generating a .pyx binding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_PYTHON_PRINT_CLASS_DEFN_HPP
#define MLPACK_BINDINGS_PYTHON_PRINT_CLASS_DEFN_HPP

#include "strip_type.hpp"

namespace mlpack {
namespace bindings {
namespace python {

/**
 * Non-serializable models don't require any special definitions, so this prints
 * nothing.
 */
template<typename T>
void PrintClassDefn(
    util::ParamData& /* d */,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0)
{
  // Do nothing.
}

/**
 * Matrices don't require any special definitions, so this prints nothing.
 */
template<typename T>
void PrintClassDefn(
    util::ParamData& /* d */,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0)
{
  // Do nothing.
}

/**
 * Serializable models require a special class definition.
 */
template<typename T>
void PrintClassDefn(
    util::ParamData& d,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0)
{
  // First, we have to parse the type.  If we have something like, e.g.,
  // 'LogisticRegression<>', we must convert this to 'LogisticRegression[].'
  std::string strippedType, printedType, defaultsType;
  StripType(d.cppType, strippedType, printedType, defaultsType);

  /**
   * This will produce code like:
   *
   * @code
   * cdef class <ModelType>Type:
   *   cdef <ModelType>* modelptr
   *   cdef public dict scrubbed_params
   *
   *   def __cinit__(self):
   *     self.modelptr = new <ModelType>()
   *     self.scrubbed_params = dict()
   *
   *   def __dealloc__(self):
   *     del self.modelptr
   *
   *   def __getstate__(self):
   *     return SerializeOut(self.modelptr, "<ModelType>")
   *
   *   def __setstate__(self, state):
   *     SerializeIn(self.modelptr, state, "<ModelType>")
   *
   *   def __reduce_ex__(self):
   *     return (self.__class__, (), self.__getstate__())
   *
   *   def _get_cpp_params(self):
   *     return SerializeOutJSON(self.modelptr, "<ModelType>")
   *
   *   def _set_cpp_params(self, state):
   *     SerializeInJSON(self.modelptr, state, "<ModelType>")
   *
   *   def get_cpp_params(self, return_str=False):
   *     params = self._get_cpp_params()
   *     return process_params_out(self, params, return_str=return_str)
   *
   *   def set_cpp_params(self, params_dic):
   *     params_str = process_params_in(self, params_dic)
   *     self._set_cpp_params(params_str)
   *
   * @endcode
   */
  std::cout << "cdef class " << strippedType << "Type:" << std::endl;
  std::cout << "  cdef " << printedType << "* modelptr" << std::endl;
  std::cout << "  cdef public dict scrubbed_params" << std::endl;
  std::cout << std::endl;
  std::cout << "  def __cinit__(self):" << std::endl;
  std::cout << "    self.modelptr = new " << printedType << "()" << std::endl;
  std::cout << "    self.scrubbed_params = dict()" << std::endl;
  std::cout << std::endl;
  std::cout << "  def __dealloc__(self):" << std::endl;
  std::cout << "    del self.modelptr" << std::endl;
  std::cout << std::endl;
  std::cout << "  def __getstate__(self):" << std::endl;
  std::cout << "    return SerializeOut(self.modelptr, \"" << printedType
      << "\")" << std::endl;
  std::cout << std::endl;
  std::cout << "  def __setstate__(self, state):" << std::endl;
  std::cout << "    SerializeIn(self.modelptr, state, \"" << printedType
      << "\")" << std::endl;
  std::cout << std::endl;
  std::cout << "  def __reduce_ex__(self, version):" << std::endl;
  std::cout << "    return (self.__class__, (), self.__getstate__())"
      << std::endl;
  std::cout << std::endl;
  std::cout << "  def _get_cpp_params(self):" << std::endl;
  std::cout << "    return SerializeOutJSON(self.modelptr, \"" << printedType
      << "\")" << std::endl;
  std::cout << std::endl;
  std::cout << "  def _set_cpp_params(self, state):" << std::endl;
  std::cout << "    SerializeInJSON(self.modelptr, state, \"" << printedType
      << "\")" << std::endl;
  std::cout << std::endl;
  std::cout << "  def get_cpp_params(self, return_str=False):" << std::endl;
  std::cout << "    params = self._get_cpp_params()" << std::endl;
  std::cout << "    return process_params_out(self, params, "
      << "return_str=return_str)" << std::endl;
  std::cout << std::endl;
  std::cout << "  def set_cpp_params(self, params_dic):" << std::endl;
  std::cout << "    params_str = process_params_in(self, params_dic)"
      << std::endl;
  std::cout << "    self._set_cpp_params(params_str.encode(\"utf-8\"))"
      << std::endl;
  std::cout << std::endl;
}

/**
 * Print the class definition to stdout.  Only serializable models require a
 * different class definition, so anything else does nothing.
 *
 * @param d Parameter data.
 * @param * (input) Unused parameter.
 * @param * (output) Unused parameter.
 */
template<typename T>
void PrintClassDefn(util::ParamData& d,
                    const void* /* input */,
                    void* /* output */)
{
  PrintClassDefn<typename std::remove_pointer<T>::type>(d);
}

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif

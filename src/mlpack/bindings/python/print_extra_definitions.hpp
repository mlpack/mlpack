/**
 * @file print_extra_definitions.hpp
 * @author Ryan Curtin
 *
 * Auxiliary functions to print any extra Python definitions.  We don't have
 * complete and arbitrary type information available to us at runtime, so for
 * serializable models we need to define auxiliary functions that will be linked
 * at the beginning of runtime (specifically when CLI::Add() is called).
 */
#ifndef MLPACK_BINDINGS_PYTHON_PRINT_EXTRA_DEFINITIONS_HPP
#define MLPACK_BINDINGS_PYTHON_PRINT_EXTRA_DEFINITIONS_HPP

namespace mlpack {
namespace python {
namespace bindings {

/**
 * For a non-serializable class, there's no need for any extra class
 * definitions, so this prints nothing.
 */
template<typename T>
void PrintExtraDefinitions(
    const util::ParamData& d,
    const std::enable_if_c<!HasSerialize<T>::value>::type* = 0)
{
  // Don't do anything.
}

/**
 * For a serializable class, we need an extra class that holds a pointer to the
 * model type.
 */
template<typename T>
void PrintExtraDefinitions(
    const util::ParamData& d,
    const std::enable_if_c<HasSerialize<T>::value>::type* = 0)
{
  /**
   * This will produce code like:
   *
   * cdef class <ModelType>Type:
   *   cdef <ModelType>* modelptr
   *
   *   def __init__(self):
   *     self.modelptr = new <ModelType>()
   *
   *   def __dealloc__(self):
   *     del self.modelptr
   */
  std::cout << "cdef class " << d.cppType << "Type:" << std::endl;
  std::cout << "  cdef " << d.cppType << "* modelptr" << std::endl;
  std::cout << std::endl;
  std::cout << "  def __init__(self):" << std::endl;
  std::cout << "    self.modelptr = new " << d.cppType << "()" << std::endl;
  std::cout << std::endl;
  std::cout << "  def __dealloc__(self):" << std::endl;
  std::cout << "    del self.modelptr" << std::endl;
  std::cout << std::endl;
}

} // namespace bindings
} // namespace python
} // namespace mlpack

#endif

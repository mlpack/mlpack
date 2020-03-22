/**
 * @file print_param_defn.hpp
 * @author Ryan Curtin
 *
 * If the type is serializable, we need to define a special utility function to
 * set a CLI parameter of that type.
 */
#ifndef MLPACK_BINDINGS_JULIA_PRINT_PARAM_DEFN_HPP
#define MLPACK_BINDINGS_JULIA_PRINT_PARAM_DEFN_HPP

#include "strip_type.hpp"

namespace mlpack {
namespace bindings {
namespace julia {

/**
 * If the type is not serializable, print nothing.
 */
template<typename T>
void PrintParamDefn(
    const util::ParamData& /* d */,
    const std::string& /* programName */,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0)
{
  // Do nothing.
}

/**
 * Matrices are serializable but here we also print nothing.
 */
template<typename T>
void PrintParamDefn(
    const util::ParamData& /* d */,
    const std::string& /* programName */,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0)
{
  // Do nothing.
}

/**
 * For non-matrix serializable types we need to print something.
 */
template<typename T>
void PrintParamDefn(
    const util::ParamData& d,
    const std::string& programName,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0)
{
  // We need to print something of the form below:
  //
  // function CLIGetParam<Type>Ptr(paramName::String)
  //   return ccall((:CLIGetParam<Type>Ptr, <programName>Library),
  //       Ptr{Nothing}, (Cstring,), paramName)
  // end
  //
  // function CLISetParam<Type>Ptr(paramName::String, ptr::Ptr{Nothing})
  //   ccall((:CLISetParam<Type>Ptr, <programName>Library), Nothing,
  //       (Cstring, Ptr{Nothing}), paramName, ptr)
  // end
  std::string type = StripType(d.cppType);
  std::cout << "\" Get the value of a model pointer parameter of type " << type
      << ".\"" << std::endl;
  std::cout << "function CLIGetParam" << type << "Ptr(paramName::String)"
      << std::endl;
  std::cout << "  return ccall((:CLI_GetParam" << type << "Ptr, "
      << programName << "Library), Ptr{Nothing}, "
      << "(Cstring,), paramName)" << std::endl;
  std::cout << "end" << std::endl;
  std::cout << std::endl;

  std::cout << "\" Set the value of a model pointer parameter of type " << type
      << ".\"" << std::endl;
  std::cout << "function CLISetParam" << type << "Ptr(paramName::String, "
      << "ptr::Ptr{Nothing})" << std::endl;
  std::cout << "  ccall((:CLI_SetParam" << type << "Ptr, "
      << programName << "Library), Nothing, (Cstring, "
      << "Ptr{Nothing}), paramName, ptr)" << std::endl;
  std::cout << "end" << std::endl;
  std::cout << std::endl;
}

/**
 * If the type is serializable, print the definition of a special utility
 * function to set a CLI parameter of that type to stdout.
 */
template<typename T>
void PrintParamDefn(const util::ParamData& d,
                    const void* input,
                    void* /* output */)
{
  PrintParamDefn<typename std::remove_pointer<T>::type>(d,
      *(std::string*) input);
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif

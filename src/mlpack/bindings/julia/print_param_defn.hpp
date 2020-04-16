/**
 * @file print_param_defn.hpp
 * @author Ryan Curtin
 *
 * If the type is serializable, we need to define a special utility function to
 * set a CLI parameter of that type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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
  // struct <Type>Ptr
  //   ptr::Ptr{Nothing}
  // end
  //
  // function CLIGetParam<Type>Ptr(paramName::String)
  //   <Type>Ptr(ccall((:CLIGetParam<Type>Ptr, <programName>Library),
  //       Ptr{Nothing}, (Cstring,), paramName))
  // end
  //
  // function CLISetParam<Type>Ptr(paramName::String, model::<Type>Ptr)
  //   ccall((:CLISetParam<Type>Ptr, <programName>Library), Nothing,
  //       (Cstring, Ptr{Nothing}), paramName, model.ptr)
  // end
  //
  // function serialize<Type>Ptr(stream::IO, model::<Type>Ptr)
  //   buffer = ccall((:Serialize<Type>Ptr, <programName>Library),
  //       Vector{UInt8}, (Ptr{Nothing},), model.ptr)
  //   write(stream, buf)
  // end
  //
  // function deserialize<Type>Ptr(stream::IO)::<Type>Ptr
  //   buffer = read(stream)
  //   <Type>Ptr(ccall((:Deserialize<Type>Ptr, <programName>Library),
  //       Ptr{Nothing}, (Vector{UInt8}, UInt), buffer, length(buffer)))
  // end

  std::string type = StripType(d.cppType);

  // First, print the struct definition.
  std::cout << "\"\"\"" << std::endl;
  std::cout << "    " << type << "Ptr" << std::endl;
  std::cout << std::endl;
  std::cout << "This type represents a C++ " << type << " model pointer.  It "
      << "can be " << std::endl << "serialized with the "
      << "`Serialization.serialize()` function." << std::endl;
  std::cout << "\"\"\"" << std::endl;
  std::cout << "struct " << type << "Ptr" << std::endl;
  std::cout << "  ptr::Ptr{Nothing}" << std::endl;
  std::cout << "end" << std::endl;
  std::cout << std::endl;

  // Now, CLIGetParam<Type>Ptr().
  std::cout << "# Get the value of a model pointer parameter of type " << type
      << "." << std::endl;
  std::cout << "function CLIGetParam" << type << "Ptr(paramName::String)::"
      << type << "Ptr" << std::endl;
  std::cout << "  " << type << "Ptr(ccall((:CLI_GetParam" << type
      << "Ptr, " << programName << "Library), Ptr{Nothing}, (Cstring,), "
      << "paramName))" << std::endl;
  std::cout << "end" << std::endl;
  std::cout << std::endl;

  // Next, CLISetParam<Type>Ptr().
  std::cout << "# Set the value of a model pointer parameter of type " << type
      << "." << std::endl;
  std::cout << "function CLISetParam" << type << "Ptr(paramName::String, "
      << "model::" << type << "Ptr)" << std::endl;
  std::cout << "  ccall((:CLI_SetParam" << type << "Ptr, "
      << programName << "Library), Nothing, (Cstring, "
      << "Ptr{Nothing}), paramName, model.ptr)" << std::endl;
  std::cout << "end" << std::endl;
  std::cout << std::endl;

  // Now the serialization functionality.
  std::cout << "# Serialize a model to the given stream." << std::endl;
  std::cout << "function serialize" << type << "Ptr(stream::IO, model::" << type
      << "Ptr)" << std::endl;
  std::cout << "  buf_len::UInt = 0" << std::endl;
  std::cout << "  buf_ptr = ccall((:Serialize" << type << "Ptr, " << programName
      << "Library), Ptr{UInt8}, (Ptr{Nothing}, Ref{UInt}), model.ptr, "
      << "Ref(buf_len))" << std::endl;
  std::cout << "  buf = Base.unsafe_wrap(buf_ptr, buf_len; own=true)"
      << std::endl;
  std::cout << "  write(stream, buf)" << std::endl;
  std::cout << "end" << std::endl;

  // And finally the deserialization functionality.
  std::cout << "# Deserialize a model from the given stream." << std::endl;
  std::cout << "function deserialize" << type << "Ptr(stream::IO)::" << type
      << "Ptr" << std::endl;
  std::cout << "  buffer = read(stream)" << std::endl;
  std::cout << "  " << type << "Ptr(ccall((:Deserialize" << type << "Ptr, "
      << programName << "Library), Ptr{Nothing}, (Vector{UInt8}, UInt), buffer,"
      << " length(buffer)))" << std::endl;
  std::cout << "end" << std::endl;
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

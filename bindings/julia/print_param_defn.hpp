/**
 * @file bindings/julia/print_param_defn.hpp
 * @author Ryan Curtin
 *
 * If the type is serializable, we need to define a special utility function to
 * set a IO parameter of that type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_JULIA_PRINT_PARAM_DEFN_HPP
#define MLPACK_BINDINGS_JULIA_PRINT_PARAM_DEFN_HPP

#include <mlpack/bindings/util/strip_type.hpp>

namespace mlpack {
namespace bindings {
namespace julia {

/**
 * If the type is not serializable, print nothing.
 */
template<typename T>
void PrintParamDefn(
    util::ParamData& /* d */,
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
    util::ParamData& /* d */,
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
    util::ParamData& d,
    const std::string& programName,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0)
{
  // We need to print something of the form below:
  //
  // import ...<Type>
  //
  // function GetParam<Type>(params::Ptr{Nothing},
  //                         paramName::String,
  //                         modelPtrs::Set{Ptr{Nothing}})
  //   ptr = ccall((:GetParam<Type>Ptr, <programName>Library),
  //       Ptr{Nothing}, (Ptr{Nothing}, Cstring,), params, paramName)
  //   return <Type>(ptr; finalize=!(ptr in modelPtrs))
  // end
  //
  // function SetParam<Type>(params::Ptr{Nothing},
  //                         paramName::String,
  //                         model::<Type>)
  //   ccall((:SetParam<Type>Ptr, <programName>Library), Nothing,
  //       (Ptr{Nothing}, Cstring, Ptr{Nothing}), params, paramName, model.ptr)
  // end
  //
  // function Delete<Type>(ptr::Ptr{Nothing})
  //   ccall((:Delete<Type>Ptr, <programName>Library), Nothing,
  //       (Ptr{Nothing},), ptr)
  // end
  //
  // function serialize<Type>(stream::IO, model::<Type>)
  //   buf_len = UInt[0]
  //   buffer = ccall((:Serialize<Type>Ptr, <programName>Library),
  //       Vector{UInt8}, (Ptr{Nothing}, Ptr{UInt8}), model.ptr,
  //       pointer(buf_len))
  //   buf = Base.unsafe_wrap(buf_ptr, buf_len[1]; own=true)
  //   write(stream, buf_len[1])
  //   write(stream, buf)
  // end
  //
  // function deserialize<Type>(stream::IO)::<Type>
  //   buf_len = read(stream, UInt)
  //   buffer = read(stream, buf_len)
  //   GC.@preserve buffer <Type>(ccall((:Deserialize<Type>Ptr,
  //       <programName>Library), Ptr{Nothing}, (Vector{UInt8}, UInt), buffer,
  //       length(buffer)))
  // end

  std::string type = util::StripType(d.cppType);

  // First, print the import of the struct.
  std::cout << "import ..." << type << std::endl;
  std::cout << std::endl;

  // Now, GetParam<Type>().
  std::cout << "# Get the value of a model pointer parameter of type " << type
      << "." << std::endl;
  std::cout << "function GetParam" << type << "(params::Ptr{Nothing}, "
      << "paramName::String, modelPtrs::Set{Ptr{Nothing}})::" << type
      << std::endl;
  std::cout << "  ptr = ccall((:GetParam" << type
      << "Ptr, " << programName << "Library), Ptr{Nothing}, (Ptr{Nothing}, "
      << "Cstring,), params, paramName)" << std::endl;
  std::cout << "  return " << type << "(ptr; finalize=!(ptr in modelPtrs))"
      << std::endl;
  std::cout << "end" << std::endl;
  std::cout << std::endl;

  // Next, IOSetParam<Type>().
  std::cout << "# Set the value of a model pointer parameter of type " << type
      << "." << std::endl;
  std::cout << "function SetParam" << type << "(params::Ptr{Nothing}, "
      << "paramName::String, model::" << type << ")" << std::endl;
  std::cout << "  ccall((:SetParam" << type << "Ptr, "
      << programName << "Library), Nothing, (Ptr{Nothing}, Cstring, "
      << "Ptr{Nothing}), params, paramName, model.ptr)" << std::endl;
  std::cout << "end" << std::endl;
  std::cout << std::endl;

  // Next, Delete<Type>().
  std::cout << "# Delete an instantiated model pointer." << std::endl;
  std::cout << "function Delete" << type << "(ptr::Ptr{Nothing})"
      << std::endl;
  std::cout << "  ccall((:Delete" << type << "Ptr, " << programName
      << "Library), Nothing, (Ptr{Nothing},), ptr)" << std::endl;
  std::cout << "end" << std::endl;
  std::cout << std::endl;

  // Now the serialization functionality.
  std::cout << "# Serialize a model to the given stream." << std::endl;
  std::cout << "function serialize" << type << "(stream::IO, model::" << type
      << ")" << std::endl;
  std::cout << "  buf_len = UInt[0]" << std::endl;
  std::cout << "  buf_ptr = ccall((:Serialize" << type << "Ptr, " << programName
      << "Library), Ptr{UInt8}, (Ptr{Nothing}, Ptr{UInt}), model.ptr, "
      << "pointer(buf_len))" << std::endl;
  std::cout << "  buf = Base.unsafe_wrap(Vector{UInt8}, buf_ptr, buf_len[1]; "
      << "own=true)" << std::endl;
  std::cout << "  write(stream, buf_len[1])" << std::endl;
  std::cout << "  write(stream, buf)" << std::endl;
  std::cout << "end" << std::endl;

  // And the deserialization functionality.
  std::cout << "# Deserialize a model from the given stream." << std::endl;
  std::cout << "function deserialize" << type << "(stream::IO)::" << type
      << std::endl;
  std::cout << "  buf_len = read(stream, UInt)" << std::endl;
  std::cout << "  buffer = read(stream, buf_len)" << std::endl;
  std::cout << "  GC.@preserve buffer " << type << "(ccall((:Deserialize"
      << type << "Ptr, " << programName << "Library), Ptr{Nothing}, "
      << "(Ptr{UInt8}, UInt), pointer(buffer), length(buffer)))" << std::endl;
  std::cout << "end" << std::endl;
}

/**
 * If the type is serializable, print the definition of a special utility
 * function to set a IO parameter of that type to stdout.
 */
template<typename T>
void PrintParamDefn(util::ParamData& d,
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

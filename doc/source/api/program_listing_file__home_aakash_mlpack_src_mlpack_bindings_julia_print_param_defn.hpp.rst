
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_julia_print_param_defn.hpp:

Program Listing for File print_param_defn.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_julia_print_param_defn.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/julia/print_param_defn.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_JULIA_PRINT_PARAM_DEFN_HPP
   #define MLPACK_BINDINGS_JULIA_PRINT_PARAM_DEFN_HPP
   
   #include <mlpack/bindings/util/strip_type.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace julia {
   
   template<typename T>
   void PrintParamDefn(
       util::ParamData& /* d */,
       const std::string& /* programName */,
       const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
       const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0)
   {
     // Do nothing.
   }
   
   template<typename T>
   void PrintParamDefn(
       util::ParamData& /* d */,
       const std::string& /* programName */,
       const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0)
   {
     // Do nothing.
   }
   
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
     // function IOGetParam<Type>(paramName::String, modelPtrs::Set{Ptr{Nothing}})
     //   ptr = ccall((:IO_GetParam<Type>Ptr, <programName>Library),
     //       Ptr{Nothing}, (Cstring,), paramName)
     //   return <Type>(ptr; finalize=!(ptr in modelPtrs))
     // end
     //
     // function IOSetParam<Type>(paramName::String, model::<Type>)
     //   ccall((:IO_SetParam<Type>Ptr, <programName>Library), Nothing,
     //       (Cstring, Ptr{Nothing}), paramName, model.ptr)
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
     //       Base.pointer(buf_len))
     //   buf = Base.unsafe_wrap(buf_ptr, buf_len[1]; own=true)
     //   write(stream, buf_len[1])
     //   write(stream, buf)
     // end
     //
     // function deserialize<Type>(stream::IO)::<Type>
     //   buf_len = read(stream, UInt)
     //   buffer = read(stream, buf_len)
     //   <Type>(ccall((:Deserialize<Type>Ptr, <programName>Library),
     //       Ptr{Nothing}, (Vector{UInt8}, UInt), buffer, length(buffer)))
     // end
   
     std::string type = util::StripType(d.cppType);
   
     // First, print the import of the struct.
     std::cout << "import ..." << type << std::endl;
     std::cout << std::endl;
   
     // Now, IOGetParam<Type>().
     std::cout << "# Get the value of a model pointer parameter of type " << type
         << "." << std::endl;
     std::cout << "function IOGetParam" << type << "(paramName::String, "
         << "modelPtrs::Set{Ptr{Nothing}})::" << type << std::endl;
     std::cout << "  ptr = ccall((:IO_GetParam" << type
         << "Ptr, " << programName << "Library), Ptr{Nothing}, (Cstring,), "
         << "paramName)" << std::endl;
     std::cout << "  return " << type << "(ptr; finalize=!(ptr in modelPtrs))"
         << std::endl;
     std::cout << "end" << std::endl;
     std::cout << std::endl;
   
     // Next, IOSetParam<Type>().
     std::cout << "# Set the value of a model pointer parameter of type " << type
         << "." << std::endl;
     std::cout << "function IOSetParam" << type << "(paramName::String, "
         << "model::" << type << ")" << std::endl;
     std::cout << "  ccall((:IO_SetParam" << type << "Ptr, "
         << programName << "Library), Nothing, (Cstring, "
         << "Ptr{Nothing}), paramName, model.ptr)" << std::endl;
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
         << "Base.pointer(buf_len))" << std::endl;
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
     std::cout << "  " << type << "(ccall((:Deserialize" << type << "Ptr, "
         << programName << "Library), Ptr{Nothing}, (Ptr{UInt8}, UInt), "
         << "Base.pointer(buffer), length(buffer)))" << std::endl;
     std::cout << "end" << std::endl;
   }
   
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

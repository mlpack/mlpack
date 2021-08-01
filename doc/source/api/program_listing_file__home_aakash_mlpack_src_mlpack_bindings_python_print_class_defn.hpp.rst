
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_python_print_class_defn.hpp:

Program Listing for File print_class_defn.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_python_print_class_defn.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/python/print_class_defn.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_PYTHON_PRINT_CLASS_DEFN_HPP
   #define MLPACK_BINDINGS_PYTHON_PRINT_CLASS_DEFN_HPP
   
   #include "strip_type.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace python {
   
   template<typename T>
   void PrintClassDefn(
       util::ParamData& /* d */,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0)
   {
     // Do nothing.
   }
   
   template<typename T>
   void PrintClassDefn(
       util::ParamData& /* d */,
       const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
   {
     // Do nothing.
   }
   
   template<typename T>
   void PrintClassDefn(
       util::ParamData& d,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
   {
     // First, we have to parse the type.  If we have something like, e.g.,
     // 'LogisticRegression<>', we must convert this to 'LogisticRegression[].'
     std::string strippedType, printedType, defaultsType;
     StripType(d.cppType, strippedType, printedType, defaultsType);
   
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
     std::cout << "    return process_params_out(self, params, return_str=return_str)" << std::endl;
     std::cout << std::endl;
     std::cout << "  def set_cpp_params(self, params_dic):" << std::endl;
     std::cout << "    params_str = process_params_in(self, params_dic)" << std::endl;
     std::cout << "    self._set_cpp_params(params_str.encode(\"utf-8\"))" << std::endl;
     std::cout << std::endl;
   }
   
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

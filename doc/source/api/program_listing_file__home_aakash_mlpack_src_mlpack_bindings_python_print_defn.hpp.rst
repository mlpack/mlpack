
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_python_print_defn.hpp:

Program Listing for File print_defn.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_python_print_defn.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/python/print_defn.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_PYTHON_PRINT_DEFN_HPP
   #define MLPACK_BINDINGS_PYTHON_PRINT_DEFN_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace python {
   
   template<typename T>
   void PrintDefn(util::ParamData& d,
                  const void* /* input */,
                  void* /* output */)
   {
     // Make sure that we don't use names that are Python keywords.
     std::string name = (d.name == "lambda") ? "lambda_" : d.name;
   
     std::cout << name;
     if (std::is_same<T, bool>::value)
       std::cout << "=False";
     else if (!d.required)
       std::cout << "=None";
   }
   
   } // namespace python
   } // namespace bindings
   } // namespace mlpack
   
   #endif

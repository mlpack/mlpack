
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_python_print_pyx.hpp:

Program Listing for File print_pyx.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_python_print_pyx.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/python/print_pyx.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_PYTHON_GENERATE_PYX_HPP
   #define MLPACK_BINDINGS_PYTHON_GENERATE_PYX_HPP
   
   #include <mlpack/core.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace python {
   
   void PrintPYX(const util::BindingDetails& doc,
                 const std::string& mainFilename,
                 const std::string& functionName);
   
   
   } // namespace python
   } // namespace bindings
   } // namespace mlpack
   
   #endif

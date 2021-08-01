
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_markdown_get_binding_name.hpp:

Program Listing for File get_binding_name.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_markdown_get_binding_name.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/markdown/get_binding_name.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_MARKDOWN_GET_BINDING_NAME_HPP
   #define MLPACK_BINDINGS_MARKDOWN_GET_BINDING_NAME_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace markdown {
   
   std::string GetBindingName(const std::string& language,
                              const std::string& name);
   
   } // namespace markdown
   } // namespace bindings
   } // namespace mlpack
   
   #endif

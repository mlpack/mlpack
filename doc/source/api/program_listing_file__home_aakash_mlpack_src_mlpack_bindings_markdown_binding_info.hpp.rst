
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_markdown_binding_info.hpp:

Program Listing for File binding_info.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_markdown_binding_info.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/markdown/binding_info.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_MARKDOWN_BINDING_NAME_HPP
   #define MLPACK_BINDINGS_MARKDOWN_BINDING_NAME_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/util/binding_details.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace markdown {
   
   class BindingInfo
   {
    public:
     static util::BindingDetails& GetBindingDetails(
         const std::string& bindingName);
   
     static std::string& Language();
   
     static BindingInfo& GetSingleton();
   
     std::unordered_map<std::string, util::BindingDetails> map;
   
    private:
     BindingInfo() { }
   
     std::string language;
   };
   
   } // namespace markdown
   } // namespace bindings
   } // namespace mlpack
   
   #endif

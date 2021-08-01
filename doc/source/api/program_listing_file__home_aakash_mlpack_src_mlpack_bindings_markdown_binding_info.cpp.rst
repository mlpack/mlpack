
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_markdown_binding_info.cpp:

Program Listing for File binding_info.cpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_markdown_binding_info.cpp>` (``/home/aakash/mlpack/src/mlpack/bindings/markdown/binding_info.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include "binding_info.hpp"
   
   using namespace std;
   
   namespace mlpack {
   namespace bindings {
   namespace markdown {
   
   util::BindingDetails& BindingInfo::GetBindingDetails(
       const std::string& bindingName)
   {
     if (GetSingleton().map.count(bindingName) == 0)
     {
       throw std::invalid_argument("Binding name '" + bindingName +
           "' not known!");
     }
   
     return GetSingleton().map.at(bindingName);
   }
   
   std::string& BindingInfo::Language()
   {
     return GetSingleton().language;
   }
   
   BindingInfo& BindingInfo::GetSingleton()
   {
     static BindingInfo instance;
     return instance;
   }
   
   } // namespace markdown
   } // namespace bindings
   } // namespace mlpack

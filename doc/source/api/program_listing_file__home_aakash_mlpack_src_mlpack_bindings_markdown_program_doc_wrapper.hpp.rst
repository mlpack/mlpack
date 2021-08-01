
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_markdown_program_doc_wrapper.hpp:

Program Listing for File program_doc_wrapper.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_markdown_program_doc_wrapper.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/markdown/program_doc_wrapper.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_MARKDOWN_PROGRAM_DOC_WRAPPER_HPP
   #define MLPACK_BINDINGS_MARKDOWN_PROGRAM_DOC_WRAPPER_HPP
   
   #include "binding_info.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace markdown {
   
   class ProgramNameWrapper
   {
    public:
     ProgramNameWrapper(const std::string& bindingName,
                        const std::string& programName)
     {
       BindingInfo::GetSingleton().map[bindingName].programName =
           std::move(programName);
     }
   };
   
   class ShortDescriptionWrapper
   {
    public:
     ShortDescriptionWrapper(const std::string& bindingName,
                             const std::string& shortDescription)
     {
       BindingInfo::GetSingleton().map[bindingName].shortDescription =
           std::move(shortDescription);
     }
   };
   
   class LongDescriptionWrapper
   {
    public:
     LongDescriptionWrapper(const std::string& bindingName,
                            const std::function<std::string()>& longDescription)
     {
       BindingInfo::GetSingleton().map[bindingName].longDescription =
           std::move(longDescription);
     }
   };
   
   class ExampleWrapper
   {
    public:
     ExampleWrapper(const std::string& bindingName,
                    const std::function<std::string()>& example)
     {
       BindingInfo::GetSingleton().map[bindingName].example.push_back(
           std::move(example));
     }
   };
   
   class SeeAlsoWrapper
   {
    public:
     SeeAlsoWrapper(const std::string& bindingName,
                    const std::string& description, const std::string& link)
     {
       BindingInfo::GetSingleton().map[bindingName].seeAlso.push_back(
           std::move(std::make_pair(description, link)));
     }
   };
   
   } // namespace markdown
   } // namespace bindings
   } // namespace mlpack
   
   #endif

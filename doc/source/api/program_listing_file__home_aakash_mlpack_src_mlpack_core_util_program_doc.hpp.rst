
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_util_program_doc.hpp:

Program Listing for File program_doc.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_util_program_doc.hpp>` (``/home/aakash/mlpack/src/mlpack/core/util/program_doc.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_UTIL_PROGRAM_DOC_HPP
   #define MLPACK_CORE_UTIL_PROGRAM_DOC_HPP
   
   namespace mlpack {
   namespace util {
   
   class ProgramName
   {
    public:
     ProgramName(const std::string& programName);
   };
   
   class ShortDescription
   {
    public:
     ShortDescription(const std::string& shortDescription);
   };
   
   class LongDescription
   {
    public:
     LongDescription(const std::function<std::string()>& longDescription);
   };
   
   class Example
   {
    public:
     Example(const std::function<std::string()>& example);
   };
   
   class SeeAlso
   {
    public:
     SeeAlso(const std::string& description, const std::string& link);
   };
   
   } // namespace util
   } // namespace mlpack
   
   #endif

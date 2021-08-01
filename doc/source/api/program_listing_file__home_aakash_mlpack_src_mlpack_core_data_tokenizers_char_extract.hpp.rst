
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_tokenizers_char_extract.hpp:

Program Listing for File char_extract.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_tokenizers_char_extract.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/tokenizers/char_extract.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_TOKENIZERS_CHAR_EXTRACT_HPP
   #define MLPACK_CORE_DATA_TOKENIZERS_CHAR_EXTRACT_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace data {
   
   class CharExtract
   {
    public:
     using TokenType = int;
   
     int operator()(boost::string_view& str) const
     {
       if (str.empty())
         return EOF;
   
       const int retval = static_cast<unsigned char>(str[0]);
   
       str.remove_prefix(1);
   
       return retval;
     }
   
     static bool IsTokenEmpty(const int token)
     {
       return token == EOF;
     }
   };
   
   } // namespace data
   } // namespace mlpack
   
   #endif


.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_tokenizers_split_by_any_of.hpp:

Program Listing for File split_by_any_of.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_tokenizers_split_by_any_of.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/tokenizers/split_by_any_of.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_TOKENIZERS_SPLIT_BY_ANY_OF_HPP
   #define MLPACK_CORE_DATA_TOKENIZERS_SPLIT_BY_ANY_OF_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/boost_backport/boost_backport_string_view.hpp>
   #include <array>
   
   namespace mlpack {
   namespace data {
   
   class SplitByAnyOf
   {
    public:
     using TokenType = boost::string_view;
   
     using MaskType = std::array<bool, 1 << CHAR_BIT>;
   
     SplitByAnyOf(const boost::string_view delimiters)
     {
       mask.fill(false);
   
       for (char symbol : delimiters)
         mask[static_cast<unsigned char>(symbol)] = true;
     }
   
     boost::string_view operator()(boost::string_view& str) const
     {
       boost::string_view retval;
   
       while (retval.empty())
       {
         const std::size_t pos = FindFirstDelimiter(str);
         if (pos == str.npos)
         {
           retval = str;
           str.clear();
           return retval;
         }
         retval = str.substr(0, pos);
         str.remove_prefix(pos + 1);
       }
       return retval;
     }
   
     static bool IsTokenEmpty(const boost::string_view token)
     {
       return token.empty();
     }
   
     const MaskType& Mask() const { return mask; }
     MaskType& Mask() { return mask; }
   
    private:
     size_t FindFirstDelimiter(const boost::string_view str) const
     {
       for (size_t pos = 0; pos < str.size(); pos++)
       {
         if (mask[static_cast<unsigned char>(str[pos])])
           return pos;
       }
       return str.npos;
     }
   
    private:
     MaskType mask;
   };
   
   } // namespace data
   } // namespace mlpack
   
   #endif

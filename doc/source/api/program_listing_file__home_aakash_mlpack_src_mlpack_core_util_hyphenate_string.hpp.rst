
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_util_hyphenate_string.hpp:

Program Listing for File hyphenate_string.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_util_hyphenate_string.hpp>` (``/home/aakash/mlpack/src/mlpack/core/util/hyphenate_string.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_UTIL_HYPHENATE_STRING_HPP
   #define MLPACK_CORE_UTIL_HYPHENATE_STRING_HPP
   
   namespace mlpack {
   namespace util {
   
   inline std::string HyphenateString(const std::string& str,
                                      const std::string& prefix,
                                      const bool force = false)
   {
     if (prefix.size() >= 80)
     {
       throw std::invalid_argument("Prefix size must be less than 80");
     }
   
     size_t margin = 80 - prefix.size();
     if (str.length() < margin && !force)
       return str;
     std::string out("");
     unsigned int pos = 0;
     // First try to look as far as possible.
     while (pos < str.length())
     {
       size_t splitpos;
       // Check that we don't have a newline first.
       splitpos = str.find('\n', pos);
       if (splitpos == std::string::npos || splitpos > (pos + margin))
       {
         // We did not find a newline.
         if (str.length() - pos < margin)
         {
           splitpos = str.length(); // The rest fits on one line.
         }
         else
         {
           splitpos = str.rfind(' ', margin + pos); // Find nearest space.
           if (splitpos <= pos || splitpos == std::string::npos) // Not found.
             splitpos = pos + margin;
         }
       }
       out += str.substr(pos, (splitpos - pos));
       if (splitpos < str.length())
       {
         out += '\n';
         out += prefix;
       }
   
       pos = splitpos;
       if (str[pos] == ' ' || str[pos] == '\n')
         pos++;
     }
     return out;
   }
   
   inline std::string HyphenateString(const std::string& str, int padding)
   {
     return HyphenateString(str, std::string(padding, ' '));
   }
   
   } // namespace util
   } // namespace mlpack
   
   #endif

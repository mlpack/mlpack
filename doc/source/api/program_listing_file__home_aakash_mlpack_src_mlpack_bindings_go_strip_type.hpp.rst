
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_go_strip_type.hpp:

Program Listing for File strip_type.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_go_strip_type.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/go/strip_type.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_GO_STRIP_TYPE_HPP
   #define MLPACK_BINDINGS_GO_STRIP_TYPE_HPP
   
   namespace mlpack {
   namespace bindings {
   namespace go {
   
   inline void StripType(const std::string& inputType,
                         std::string& goStrippedType,
                         std::string& strippedType,
                         std::string& printedType,
                         std::string& defaultsType)
   {
     // First, we have to parse the type.  If we have something like, e.g.,
     // 'LogisticRegression<>', we must convert this to 'LogisticRegression<T>.'
     goStrippedType = inputType;
     printedType = inputType;
     strippedType = inputType;
     defaultsType = inputType;
   
     // Lower the first letter of parameter name so it is
     // of unexported type in GoDoc.
     for (size_t i = 0; i < goStrippedType.size(); i++)
     {
       goStrippedType[i] = std::tolower(goStrippedType[i]);
       if (std::isupper(goStrippedType[i+1]) && !std::isupper(goStrippedType[i+2]))
       {
         if (isalpha(goStrippedType[i+2]))
           break;
       }
     }
   
     if (printedType.find("<") != std::string::npos)
     {
       // Are there any template parameters?  Or is it the default?
       const size_t loc = printedType.find("<>");
       if (loc != std::string::npos)
       {
         // Convert it from "<>".
         goStrippedType.replace(loc, 2, "");
         strippedType.replace(loc, 2, "");
         printedType.replace(loc, 2, "<>");
         defaultsType.replace(loc, 2, "<T>");
       }
     }
   }
   
   } // namespace go
   } // namespace bindings
   } // namespace mlpack
   
   #endif

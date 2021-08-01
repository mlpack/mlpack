
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_markdown_get_binding_name.cpp:

Program Listing for File get_binding_name.cpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_markdown_get_binding_name.cpp>` (``/home/aakash/mlpack/src/mlpack/bindings/markdown/get_binding_name.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include "get_binding_name.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace markdown {
   
   std::string GetBindingName(const std::string& language,
                              const std::string& name)
   {
     // Unfortunately, every time a new binding is added, this code will need to be
     // modified.
     if (language == "cli")
     {
       // For command-line programs, all bindings have 'mlpack_' prepended to the
       // name.
       return "mlpack_" + name;
     }
     else if (language == "python")
     {
       // For Python bindings, the name is unchanged.
       return name;
     }
     else if (language == "julia")
     {
       // For Julia bindings, the name is unchanged.
       return name;
     }
     else if (language == "go")
     {
       // For Go bindings, the name is unchanged.
       return name;
     }
     else if (language == "r")
     {
       // For R bindings, the name is unchanged.
       return name;
     }
     else
     {
       throw std::invalid_argument("Don't know how to compute binding name for "
           "language \"" + language + "\"!  Is the language specified in "
           "src/mlpack/bindings/markdown/get_binding_name.cpp?");
     }
   }
   
   } // namespace markdown
   } // namespace bindings
   } // namespace mlpack

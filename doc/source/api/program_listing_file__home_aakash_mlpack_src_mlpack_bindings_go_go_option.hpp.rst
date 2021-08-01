
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_go_go_option.hpp:

Program Listing for File go_option.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_go_go_option.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/go/go_option.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_GOLANG_GO_OPTION_HPP
   #define MLPACK_BINDINGS_GOLANG_GO_OPTION_HPP
   
   #include <mlpack/core/util/param_data.hpp>
   #include "get_param.hpp"
   #include "get_type.hpp"
   #include "default_param.hpp"
   #include "get_printable_param.hpp"
   #include "print_defn_input.hpp"
   #include "print_defn_output.hpp"
   #include "print_doc.hpp"
   #include "print_input_processing.hpp"
   #include "print_method_config.hpp"
   #include "print_method_init.hpp"
   #include "print_output_processing.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace go {
   
   // Defined in mlpack_main.hpp.
   extern std::string programName;
   
   template<typename T>
   class GoOption
   {
    public:
     GoOption(const T defaultValue,
              const std::string& identifier,
              const std::string& description,
              const std::string& alias,
              const std::string& cppName,
              const bool required = false,
              const bool input = true,
              const bool noTranspose = false,
              const std::string& /*testName*/ = "")
     {
       // Create the ParamData object to give to IO.
       util::ParamData data;
   
       data.desc = description;
       data.name = identifier;
       data.tname = TYPENAME(T);
       data.alias = alias[0];
       data.wasPassed = false;
       data.noTranspose = noTranspose;
       data.required = required;
       data.input = input;
       data.loaded = false;
       // Only "verbose" and "copy_all_inputs" will be persistent.
       if (identifier == "verbose" /*|| identifier == "copy_all_inputs"*/)
         data.persistent = true;
       else
         data.persistent = false;
       data.cppType = cppName;
   
       data.value = boost::any(defaultValue);
   
       // Restore the parameters for this program.
       if (identifier != "verbose" /*&& identifier != "copy_all_inputs"*/)
         IO::RestoreSettings(programName, false);
   
       // Set the function pointers that we'll need.  All of these function
       // pointers will be used by both the program that generates the .cpp,
       // the .h, and the .go binding files.
       IO::GetSingleton().functionMap[data.tname]["GetParam"] = &GetParam<T>;
       IO::GetSingleton().functionMap[data.tname]["GetPrintableParam"] =
           &GetPrintableParam<T>;
   
       IO::GetSingleton().functionMap[data.tname]["DefaultParam"] =
           &DefaultParam<T>;
       IO::GetSingleton().functionMap[data.tname]["PrintDefnInput"] =
           &PrintDefnInput<T>;
       IO::GetSingleton().functionMap[data.tname]["PrintDefnOutput"] =
           &PrintDefnOutput<T>;
       IO::GetSingleton().functionMap[data.tname]["PrintDoc"] = &PrintDoc<T>;
       IO::GetSingleton().functionMap[data.tname]["PrintOutputProcessing"] =
           &PrintOutputProcessing<T>;
       IO::GetSingleton().functionMap[data.tname]["PrintMethodConfig"] =
           &PrintMethodConfig<T>;
       IO::GetSingleton().functionMap[data.tname]["PrintMethodInit"] =
           &PrintMethodInit<T>;
       IO::GetSingleton().functionMap[data.tname]["PrintInputProcessing"] =
           &PrintInputProcessing<T>;
       IO::GetSingleton().functionMap[data.tname]["GetType"] = &GetType<T>;
   
       // Add the ParamData object, then store.  This is necessary because we may
       // import more than one .so that uses IO, so we have to keep the options
       // separate.  programName is a global variable from mlpack_main.hpp.
       IO::Add(std::move(data));
       if (identifier != "verbose" /*&& identifier != "copy_all_inputs"*/)
         IO::StoreSettings(programName);
       IO::ClearSettings();
     }
   };
   
   } // namespace go
   } // namespace bindings
   } // namespace mlpack
   
   #endif

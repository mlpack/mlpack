
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_cli_cli_option.hpp:

Program Listing for File cli_option.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_cli_cli_option.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/cli/cli_option.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_BINDINGS_CLI_CLI_OPTION_HPP
   #define MLPACK_CORE_BINDINGS_CLI_CLI_OPTION_HPP
   
   #include <string>
   
   #include <mlpack/core/util/io.hpp>
   #include "parameter_type.hpp"
   #include "add_to_cli11.hpp"
   #include "default_param.hpp"
   #include "output_param.hpp"
   #include "get_printable_param.hpp"
   #include "string_type_param.hpp"
   #include "get_param.hpp"
   #include "get_raw_param.hpp"
   #include "map_parameter_name.hpp"
   #include "set_param.hpp"
   #include "get_printable_param_name.hpp"
   #include "get_printable_param_value.hpp"
   #include "get_allocated_memory.hpp"
   #include "delete_allocated_memory.hpp"
   #include "in_place_copy.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace cli {
   
   template<typename N>
   class CLIOption
   {
    public:
     CLIOption(const N defaultValue,
               const std::string& identifier,
               const std::string& description,
               const std::string& alias,
               const std::string& cppName,
               const bool required = false,
               const bool input = true,
               const bool noTranspose = false,
               const std::string& /*testName*/ = "")
     {
       // Create the ParamData object to give to CLI.
       util::ParamData data;
   
       data.desc = description;
       data.name = identifier;
       data.tname = TYPENAME(N);
       data.alias = alias[0];
       data.wasPassed = false;
       data.noTranspose = noTranspose;
       data.required = required;
       data.input = input;
       data.loaded = false;
       data.persistent = false; // All CLI parameters are not persistent.
       data.cppType = cppName;
   
       // Apply default value.
       if (std::is_same<typename std::remove_pointer<N>::type,
                        typename ParameterType<typename
                            std::remove_pointer<N>::type>::type>::value)
       {
         data.value = boost::any(defaultValue);
       }
       else
       {
         typename ParameterType<typename std::remove_pointer<N>::type>::type tmp;
         data.value = boost::any(std::tuple<N, decltype(tmp)>(defaultValue, tmp));
       }
   
       const std::string tname = data.tname;
       const std::string cliName = MapParameterName<
           typename std::remove_pointer<N>::type>(identifier);
       std::string progOptId = (alias[0] != '\0') ?
           "-" + std::string(1, alias[0]) + ",--" + cliName : "--" + cliName;
   
       // Do a check to ensure that the boost name isn't already in use.
       const std::map<std::string, util::ParamData>& parameters =
           IO::Parameters();
       if (parameters.count(cliName) > 0)
       {
         // Create a fake Log::Fatal since it may not yet be initialized.
         // Temporarily define color code escape sequences.
         #ifndef _WIN32
           #define BASH_RED "\033[0;31m"
           #define BASH_CLEAR "\033[0m"
         #else
           #define BASH_RED ""
           #define BASH_CLEAR ""
         #endif
   
         // Temporary outstream object for detecting duplicate identifiers.
         util::PrefixedOutStream outstr(std::cerr,
               BASH_RED "[FATAL] " BASH_CLEAR, false, true /* fatal */);
   
         #undef BASH_RED
         #undef BASH_CLEAR
   
         outstr << "Parameter --" << cliName << " (" << data.alias << ") "
                << "is defined multiple times with the same identifiers."
                << std::endl;
       }
   
       IO::Add(std::move(data));
   
       // Set some function pointers that we need.
       IO::GetSingleton().functionMap[tname]["DefaultParam"] =
           &DefaultParam<N>;
       IO::GetSingleton().functionMap[tname]["OutputParam"] =
           &OutputParam<N>;
       IO::GetSingleton().functionMap[tname]["GetPrintableParam"] =
           &GetPrintableParam<N>;
       IO::GetSingleton().functionMap[tname]["StringTypeParam"] =
           &StringTypeParam<N>;
       IO::GetSingleton().functionMap[tname]["GetParam"] = &GetParam<N>;
       IO::GetSingleton().functionMap[tname]["GetRawParam"] = &GetRawParam<N>;
       IO::GetSingleton().functionMap[tname]["AddToCLI11"] = &AddToCLI11<N>;
       IO::GetSingleton().functionMap[tname]["MapParameterName"] =
           &MapParameterName<N>;
       IO::GetSingleton().functionMap[tname]["GetPrintableParamName"] =
           &GetPrintableParamName<N>;
       IO::GetSingleton().functionMap[tname]["GetPrintableParamValue"] =
           &GetPrintableParamValue<N>;
       IO::GetSingleton().functionMap[tname]["GetAllocatedMemory"] =
           &GetAllocatedMemory<N>;
       IO::GetSingleton().functionMap[tname]["DeleteAllocatedMemory"] =
           &DeleteAllocatedMemory<N>;
       IO::GetSingleton().functionMap[tname]["InPlaceCopy"] = &InPlaceCopy<N>;
     }
   };
   
   } // namespace cli
   } // namespace bindings
   } // namespace mlpack
   
   #endif

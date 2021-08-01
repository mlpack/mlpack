
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_cli_print_doc_functions_impl.hpp:

Program Listing for File print_doc_functions_impl.hpp
=====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_cli_print_doc_functions_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/cli/print_doc_functions_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_CLI_PRINT_DOC_FUNCTIONS_IMPL_HPP
   #define MLPACK_BINDINGS_CLI_PRINT_DOC_FUNCTIONS_IMPL_HPP
   
   #include <mlpack/core/util/hyphenate_string.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace cli {
   
   inline std::string GetBindingName(const std::string& bindingName)
   {
     return "mlpack_" + bindingName;
   }
   
   inline std::string PrintImport(const std::string& /* bindingName */)
   {
     return "";
   }
   
   inline std::string PrintInputOptionInfo()
   {
     return "";
   }
   inline std::string PrintOutputOptionInfo()
   {
     return "";
   }
   
   template<typename T>
   inline std::string PrintValue(const T& value, bool quotes)
   {
     std::ostringstream oss;
     if (quotes)
       oss << "'";
     oss << value;
     if (quotes)
       oss << "'";
     return oss.str();
   }
   
   template<typename T>
   inline std::string PrintValue(const std::vector<T>& value, bool quotes)
   {
     std::ostringstream oss;
     if (quotes)
       oss << "'";
     if (value.size() > 0)
     {
       oss << value[0];
       for (size_t i = 1; i < value.size(); ++i)
         oss << ", " << value[i];
     }
     if (quotes)
       oss << "'";
     return oss.str();
   }
   
   inline std::string PrintDefault(const std::string& paramName)
   {
     if (IO::Parameters().count(paramName) == 0)
       throw std::invalid_argument("unknown parameter " + paramName + "!");
   
     util::ParamData& d = IO::Parameters()[paramName];
   
     std::string defaultValue;
     IO::GetSingleton().functionMap[d.tname]["DefaultParam"](d, NULL,
         (void*) &defaultValue);
   
     return defaultValue;
   }
   
   inline std::string PrintDataset(const std::string& dataset)
   {
     return "'" + dataset + ".csv'";
   }
   
   inline std::string PrintModel(const std::string& model)
   {
     return "'" + model + ".bin'";
   }
   
   // Base case for recursion.
   inline std::string ProcessOptions() { return ""; }
   
   template<typename T, typename... Args>
   std::string ProcessOptions(const std::string& paramName,
                              const T& value,
                              Args... args)
   {
     // See if it is part of the program.
     std::string result = "";
     if (IO::Parameters().count(paramName) > 0)
     {
       util::ParamData& d = IO::Parameters()[paramName];
   
       std::string name;
       IO::GetSingleton().functionMap[d.tname]["GetPrintableParamName"](d, NULL,
           (void*) &name);
   
       std::ostringstream ossValue;
       ossValue << value;
       std::string rawValue = ossValue.str();
       std::string fullValue;
       IO::GetSingleton().functionMap[d.tname]["GetPrintableParamValue"](d,
           (void*) &rawValue, (void*) &fullValue);
   
       std::ostringstream oss;
       if (d.tname != TYPENAME(bool))
         oss << name << " " << fullValue;
       else
         oss << name;
       result = oss.str();
     }
     else
     {
       throw std::runtime_error("Unknown parameter '" + paramName + "' " +
           "encountered while assembling documentation!  Check BINDING_LONG_DESC()"
           + " and BINDING_EXAMPLE() declaration.");
     }
   
     std::string rest = ProcessOptions(args...);
     if (rest != "")
       result += " " + rest;
   
     return result;
   }
   
   template<typename... Args>
   std::string ProgramCall(const std::string& programName, Args... args)
   {
     return util::HyphenateString("$ " + GetBindingName(programName) + " " +
         ProcessOptions(args...), 2);
   }
   
   inline std::string ProgramCall(const std::string& programName)
   {
     std::ostringstream oss;
     oss << "$ " << GetBindingName(programName);
   
     // Handle all options---first input options, then output options.
     std::map<std::string, util::ParamData>& parameters = IO::Parameters();
   
     for (auto& it : parameters)
     {
       if (!it.second.input || it.second.persistent)
         continue;
   
       // Otherwise, print the name and the default value.
       std::string name;
       IO::GetSingleton().functionMap[it.second.tname]["GetPrintableParamName"](
           it.second, NULL, (void*) &name);
   
       std::string value;
       IO::GetSingleton().functionMap[it.second.tname]["DefaultParam"](
           it.second, NULL, (void*) &value);
       if (value == "''")
         value = "<string>";
   
       oss << " ";
       if (!it.second.required)
         oss << "[";
   
       oss << name;
       if (it.second.cppType != "bool")
         oss << " " << value;
   
       if (!it.second.required)
         oss << "]";
     }
   
     // Now get the output options.
     for (auto& it : parameters)
     {
       if (it.second.input)
         continue;
   
       // Otherwise, print the name and the default value.
       std::string name;
       IO::GetSingleton().functionMap[it.second.tname]["GetPrintableParamName"](
           it.second, NULL, (void*) &name);
   
       std::string value;
       IO::GetSingleton().functionMap[it.second.tname]["DefaultParam"](
           it.second, NULL, (void*) &value);
       if (value == "''")
         value = "<string>";
   
       oss << " [" << name;
       if (it.second.cppType != "bool")
         oss << " " << value;
       oss << "]";
     }
   
     return util::HyphenateString(oss.str(), 8);
   }
   
   inline std::string ParamString(const std::string& paramName)
   {
     // Return the correct parameter name.
     if (IO::Parameters().count(paramName) > 0)
     {
       util::ParamData& d = IO::Parameters()[paramName];
   
       std::string output;
       IO::GetSingleton().functionMap[d.tname]["GetPrintableParamName"](d, NULL,
           (void*) &output);
       // Is there an alias?
       std::string alias = "";
       if (d.alias != '\0')
         alias = " (-" + std::string(1, d.alias) + ")";
   
       return "'" + output + alias + "'";
     }
     else
     {
       throw std::runtime_error("Parameter '" + paramName + "' not known!  Check "
           "BINDING_LONG_DESC() and BINDING_EXAMPLE() definition.");
     }
   }
   
   } // namespace cli
   } // namespace bindings
   } // namespace mlpack
   
   #endif


.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_cli_print_help.cpp:

Program Listing for File print_help.cpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_cli_print_help.cpp>` (``/home/aakash/mlpack/src/mlpack/bindings/cli/print_help.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include "print_help.hpp"
   
   #include <mlpack/core.hpp>
   #include <mlpack/core/util/hyphenate_string.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace cli {
   
   /* Prints the descriptions of the current hierarchy. */
   void PrintHelp(const std::string& param)
   {
     std::string usedParam = param;
     std::map<std::string, util::ParamData>& parameters = IO::Parameters();
     const std::map<char, std::string>& aliases = IO::Aliases();
     util::BindingDetails& bindingDetails = IO::GetSingleton().doc;
     // If we pass a single param, alias it if necessary.
     if (usedParam.length() == 1 && aliases.count(usedParam[0]))
       usedParam = aliases.at(usedParam[0]);
   
     // Do we only want to print out one value?
     if (usedParam != "" && parameters.count(usedParam))
     {
       util::ParamData& data = parameters.at(usedParam);
       std::string alias = (data.alias != '\0') ? " (-"
           + std::string(1, data.alias) + ")" : "";
   
       // Figure out the name of the type.
       std::string printableType;
       IO::GetSingleton().functionMap[data.tname]["StringTypeParam"](data, NULL,
           (void*) &printableType);
       std::string type = " [" + printableType + "]";
   
       // Now, print the descriptions.
       std::string fullDesc = "  --" + usedParam + alias + type + "  ";
   
       if (fullDesc.length() <= 32) // It all fits on one line.
         std::cout << fullDesc << std::string(32 - fullDesc.length(), ' ');
       else // We need multiple lines.
         std::cout << fullDesc << std::endl << std::string(32, ' ');
   
       std::cout << util::HyphenateString(data.desc, 32) << std::endl;
       return;
     }
     else if (usedParam != "")
     {
       // User passed a single variable, but it doesn't exist.
       std::cerr << "Parameter --" << usedParam << " does not exist."
           << std::endl;
       exit(1); // Nothing left to do.
     }
   
     // Print out the descriptions.
     if (bindingDetails.programName != "")
     {
       std::cout << bindingDetails.programName << std::endl << std::endl;
       std::cout << "  " << util::HyphenateString(bindingDetails.longDescription(),
           2) << std::endl << std::endl;
       for (size_t j = 0; j < bindingDetails.example.size(); ++j)
       {
         std::cout << "  " << util::HyphenateString(bindingDetails.example[j](), 2)
             << std::endl << std::endl;
       }
     }
     else
       std::cout << "[undocumented program]" << std::endl << std::endl;
   
     for (size_t pass = 0; pass < 3; ++pass)
     {
       bool printedHeader = false;
       // Print out the descriptions of everything else.
       for (auto& iter : parameters)
       {
         util::ParamData& data = iter.second;
         const std::string key;
         IO::GetSingleton().functionMap[data.tname]["MapParameterName"](data,
             NULL, (void*) &key);
   
         std::string desc = data.desc;
         std::string alias = (iter.second.alias != '\0') ?
             std::string(1, iter.second.alias) : "";
         alias = alias.length() ? " (-" + alias + ")" : alias;
   
         // Filter un-printed options.
         if ((pass == 0) && !(data.required && data.input)) // Required input.
           continue;
         if ((pass == 1) && !(!data.required && data.input)) // Optional input.
           continue;
         if ((pass == 2) && data.input) // Output options only (always optional).
           continue;
   
         // For reverse compatibility: this can be removed when these options are
         // gone in mlpack 3.0.0.  We don't want to print the deprecated options.
         if (data.name == "inputFile")
           continue;
   
         if (!printedHeader)
         {
           printedHeader = true;
           if (pass == 0)
             std::cout << "Required input options:" << std::endl << std::endl;
           else if (pass == 1)
             std::cout << "Optional input options: " << std::endl << std::endl;
           else if (pass == 2)
             std::cout << "Optional output options: " << std::endl << std::endl;
         }
   
         // Append default value to description.
         if (pass >= 1 && (data.cppType == "int" || data.cppType == "double" ||
                           data.cppType == "std::string" ||
                           data.cppType == "std::vector<int>" ||
                           data.cppType == "std::vector<double>" ||
                           data.cppType == "std::vector<std::string>"))
         {
           std::string defaultValue;
           IO::GetSingleton().functionMap[data.tname]["DefaultParam"](data,
               NULL, (void*) &defaultValue);
           desc += "  Default value " + defaultValue + ".";
         }
   
         // Now, print the descriptions.
         std::string printableType;
         IO::GetSingleton().functionMap[data.tname]["StringTypeParam"](data,
             NULL, (void*) &printableType);
         std::string type = " [" + printableType + "]";
         std::string fullDesc = "  --" + key + alias + type + "  ";
   
         if (fullDesc.length() <= 32) // It all fits on one line.
           std::cout << fullDesc << std::string(32 - fullDesc.length(), ' ');
         else // We need multiple lines.
           std::cout << fullDesc << std::endl << std::string(32, ' ');
   
         std::cout << util::HyphenateString(desc, 32) << std::endl;
       }
   
       if (printedHeader)
         std::cout << std::endl;
     }
   
     // Helpful information at the bottom of the help output, to point the user to
     // citations and better documentation (if necessary).  See ticket #195.
     std::cout << util::HyphenateString("For further information, including "
         "relevant papers, citations, and theory, consult the documentation found "
         "at http://www.mlpack.org or included with your distribution of mlpack.",
         0) << std::endl;
   }
   
   
   } // namespace cli
   } // namespace bindings
   } // namespace mlpack

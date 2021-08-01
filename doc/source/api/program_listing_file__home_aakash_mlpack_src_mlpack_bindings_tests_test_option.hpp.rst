
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_tests_test_option.hpp:

Program Listing for File test_option.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_tests_test_option.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/tests/test_option.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_BINDINGS_TESTS_TEST_OPTION_HPP
   #define MLPACK_CORE_BINDINGS_TESTS_TEST_OPTION_HPP
   
   #include <string>
   
   #include <mlpack/core/util/io.hpp>
   #include "get_printable_param.hpp"
   #include "get_param.hpp"
   #include "get_allocated_memory.hpp"
   #include "delete_allocated_memory.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace tests {
   
   // Defined in mlpack_main.hpp.
   extern std::string programName;
   
   template<typename N>
   class TestOption
   {
    public:
     TestOption(const N defaultValue,
                const std::string& identifier,
                const std::string& description,
                const std::string& alias,
                const std::string& cppName,
                const bool required = false,
                const bool input = true,
                const bool noTranspose = false,
                const std::string& testName = "")
     {
       // Create the ParamData object to give to IO.
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
       data.cppType = cppName;
       data.value = boost::any(defaultValue);
       data.persistent = false;
   
       const std::string tname = data.tname;
   
       IO::RestoreSettings(testName, false);
   
       // Set some function pointers that we need.
       IO::GetSingleton().functionMap[tname]["GetPrintableParam"] =
           &GetPrintableParam<N>;
       IO::GetSingleton().functionMap[tname]["GetParam"] = &GetParam<N>;
       IO::GetSingleton().functionMap[tname]["GetAllocatedMemory"] =
           &GetAllocatedMemory<N>;
       IO::GetSingleton().functionMap[tname]["DeleteAllocatedMemory"] =
           &DeleteAllocatedMemory<N>;
   
       IO::Add(std::move(data));
   
       // If this is an output option, set it as passed.
       if (!input)
         IO::SetPassed(identifier);
   
       IO::StoreSettings(testName);
       IO::ClearSettings();
     }
   };
   
   } // namespace tests
   } // namespace bindings
   } // namespace mlpack
   
   #endif

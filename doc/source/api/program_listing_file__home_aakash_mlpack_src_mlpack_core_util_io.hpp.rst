
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_util_io.hpp:

Program Listing for File io.hpp
===============================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_util_io.hpp>` (``/home/aakash/mlpack/src/mlpack/core/util/io.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_UTIL_IO_HPP
   #define MLPACK_CORE_UTIL_IO_HPP
   
   #include <iostream>
   #include <list>
   #include <map>
   #include <string>
   
   #include <boost/any.hpp>
   
   #include <mlpack/prereqs.hpp>
   
   #include "timers.hpp"
   #include "binding_details.hpp"
   #include "program_doc.hpp"
   #include "version.hpp"
   
   #include "param_data.hpp"
   
   #include <mlpack/core/data/load.hpp>
   #include <mlpack/core/data/save.hpp>
   
   namespace mlpack {
   
   class IO
   {
    public:
     static void Add(util::ParamData&& d);
   
     static bool HasParam(const std::string& identifier);
   
     template<typename T>
     static T& GetParam(const std::string& identifier);
   
     template<typename T>
     static std::string GetPrintableParam(const std::string& identifier);
   
     template<typename T>
     static T& GetRawParam(const std::string& identifier);
   
     template<typename T>
     static void CheckInputMatrix(const T& matrix, const std::string& identifier);
   
     static void MakeInPlaceCopy(const std::string& outputParamName,
                                 const std::string& inputParamName);
   
     static IO& GetSingleton();
   
     static std::map<std::string, util::ParamData>& Parameters();
     static std::map<char, std::string>& Aliases();
   
     static std::string ProgramName();
   
     static void SetPassed(const std::string& name);
   
     static void StoreSettings(const std::string& name);
   
     static void RestoreSettings(const std::string& name, const bool fatal = true);
   
     static void ClearSettings();
   
     static void CheckInputMatrices();
   
    private:
     std::map<char, std::string> aliases;
     std::map<std::string, util::ParamData> parameters;
   
    public:
     typedef std::map<std::string, std::map<std::string,
         void (*)(util::ParamData&, const void*, void*)>> FunctionMapType;
     FunctionMapType functionMap;
   
    private:
     std::map<std::string, std::tuple<std::map<std::string, util::ParamData>,
         std::map<char, std::string>, FunctionMapType>> storageMap;
   
    public:
     bool didParse;
   
     std::string programName;
   
     Timers timer;
   
     friend class Timer;
   
     util::BindingDetails doc;
    private:
     IO();
   
     IO(const IO& other);
     IO& operator=(const IO& other);
   };
   
   } // namespace mlpack
   
   // Include the actual definitions of templated methods
   #include "io_impl.hpp"
   
   #endif

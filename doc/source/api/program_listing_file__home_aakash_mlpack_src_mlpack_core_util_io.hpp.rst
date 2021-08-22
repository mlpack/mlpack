
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
   #include "params.hpp"
   
   #include <mlpack/core/data/load.hpp>
   #include <mlpack/core/data/save.hpp>
   
   // TODO: this entire set of code is related to the bindings and maybe should go
   // into src/mlpack/bindings/util/.
   namespace mlpack {
   
   // TODO: completely go through this documentation and clean it up
   class IO
   {
    public:
     static void AddParameter(const std::string& bindingName, util::ParamData&& d);
   
     static void AddFunction(const std::string& type,
                             const std::string& name,
                             void (*func)(util::ParamData&, const void*, void*));
   
     static void AddBindingName(const std::string& bindingName,
                                const std::string& name);
   
     static void AddShortDescription(const std::string& bindingName,
                                     const std::string& shortDescription);
   
     static void AddLongDescription(
         const std::string& bindingName,
         const std::function<std::string()>& longDescription);
   
     static void AddExample(const std::string& bindingName,
                            const std::function<std::string()>& example);
   
     static void AddSeeAlso(const std::string& bindingName,
                            const std::string& description,
                            const std::string& link);
   
     static util::Params Parameters(const std::string& bindingName);
   
     static IO& GetSingleton();
   
     static util::Timers& GetTimers();
   
    private:
     std::mutex mapMutex;
     std::map<std::string, std::map<char, std::string>> aliases;
     std::map<std::string, std::map<std::string, util::ParamData>> parameters;
     typedef std::map<std::string, std::map<std::string,
         void (*)(util::ParamData&, const void*, void*)>> FunctionMapType;
     FunctionMapType functionMap;
   
     std::mutex docMutex;
     std::map<std::string, util::BindingDetails> docs;
   
     util::Timers timer;
   
     friend class Timer;
   
     IO();
   
     IO(const IO& other);
     IO& operator=(const IO& other);
   };
   
   } // namespace mlpack
   
   // Include the actual definitions of templated methods
   #include "io_impl.hpp"
   
   #endif

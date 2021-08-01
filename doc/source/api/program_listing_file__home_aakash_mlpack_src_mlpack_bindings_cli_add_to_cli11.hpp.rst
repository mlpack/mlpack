
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_cli_add_to_cli11.hpp:

Program Listing for File add_to_cli11.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_cli_add_to_cli11.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/cli/add_to_cli11.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_CLI_ADD_TO_CLI11_HPP
   #define MLPACK_BINDINGS_CLI_ADD_TO_CLI11_HPP
   
   #include <mlpack/core/util/param_data.hpp>
   #include <mlpack/core/util/is_std_vector.hpp>
   #include "map_parameter_name.hpp"
   
   #include "third_party/CLI/CLI11.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace cli {
   
   template<typename T>
   void AddToCLI11(const std::string& cliName,
                   util::ParamData& param,
                   CLI::App& app,
                   const typename boost::disable_if<std::is_same<T,
                       bool>>::type* = 0,
                   const typename boost::disable_if<
                       arma::is_arma_type<T>>::type* = 0,
                   const typename boost::disable_if<
                       data::HasSerialize<T>>::type* = 0,
                   const typename boost::enable_if<std::is_same<T,
                       std::tuple<mlpack::data::DatasetInfo,
                       arma::mat>>>::type* = 0)
   {
     app.add_option_function<std::string>(cliName.c_str(),
         [&param](const std::string& value)
         {
           using TupleType = std::tuple<T, typename ParameterType<T>::type>;
           TupleType& tuple = *boost::any_cast<TupleType>(&param.value);
           std::get<0>(std::get<1>(tuple)) = boost::any_cast<std::string>(value);
           param.wasPassed = true;
         },
         param.desc.c_str());
   }
   
   template<typename T>
   void AddToCLI11(const std::string& cliName,
                   util::ParamData& param,
                   CLI::App& app,
                   const typename boost::disable_if<std::is_same<T,
                       bool>>::type* = 0,
                   const typename boost::disable_if<
                       arma::is_arma_type<T>>::type* = 0,
                   const typename boost::enable_if<
                       data::HasSerialize<T>>::type* = 0,
                   const typename boost::disable_if<std::is_same<T,
                       std::tuple<mlpack::data::DatasetInfo,
                       arma::mat>>>::type* = 0)
   {
     app.add_option_function<std::string>(cliName.c_str(),
         [&param](const std::string& value)
         {
           using TupleType = std::tuple<T*, typename ParameterType<T>::type>;
           TupleType& tuple = *boost::any_cast<TupleType>(&param.value);
           std::get<1>(tuple) = boost::any_cast<std::string>(value);
           param.wasPassed = true;
         },
         param.desc.c_str());
   }
   
   template<typename T>
   void AddToCLI11(const std::string& cliName,
                   util::ParamData& param,
                   CLI::App& app,
                   const typename boost::disable_if<
                       std::is_same<T, bool>>::type* = 0,
                   const typename boost::enable_if<
                       arma::is_arma_type<T>>::type* = 0,
                   const typename boost::disable_if<std::is_same<T,
                     std::tuple<mlpack::data::DatasetInfo,
                       arma::mat>>>::type* = 0)
   {
     app.add_option_function<std::string>(cliName.c_str(),
         [&param](const std::string& value)
         {
           using TupleType = std::tuple<T, typename ParameterType<T>::type>;
           TupleType& tuple = *boost::any_cast<TupleType>(&param.value);
           std::get<0>(std::get<1>(tuple)) = boost::any_cast<std::string>(value);
           param.wasPassed = true;
         },
         param.desc.c_str());
   }
   
   template<typename T>
   void AddToCLI11(const std::string& cliName,
                   util::ParamData& param,
                   CLI::App& app,
                   const typename boost::disable_if<
                       std::is_same<T, bool>>::type* = 0,
                   const typename boost::disable_if<
                       arma::is_arma_type<T>>::type* = 0,
                   const typename boost::disable_if<
                       data::HasSerialize<T>>::type* = 0,
                   const typename boost::disable_if<std::is_same<T,
                       std::tuple<mlpack::data::DatasetInfo,
                       arma::mat>>>::type* = 0)
   {
     app.add_option_function<T>(cliName.c_str(),
         [&param](const T& value)
         {
           param.value = value;
           param.wasPassed = true;
         },
         param.desc.c_str());
   }
   
   template<typename T>
   void AddToCLI11(const std::string& cliName,
                   util::ParamData& param,
                   CLI::App& app,
                   const typename boost::enable_if<
                       std::is_same<T, bool>>::type* = 0,
                   const typename boost::disable_if<
                       arma::is_arma_type<T>>::type* = 0,
                   const typename boost::disable_if<
                       data::HasSerialize<T>>::type* = 0,
                   const typename boost::disable_if<std::is_same<T,
                       std::tuple<mlpack::data::DatasetInfo,
                       arma::mat>>>::type* = 0)
   {
     app.add_flag_function(cliName.c_str(),
         [&param](const T& value)
         {
           param.value = value;
           param.wasPassed = true;
         },
         param.desc.c_str());
   }
   
   template<typename T>
   void AddToCLI11(util::ParamData& param,
                   const void* /* input */,
                   void* output)
   {
     // Cast CLI::App object.
     CLI::App* app = (CLI::App*) output;
   
     // Generate the name to be given to CLI11.
     const std::string mappedName =
         MapParameterName<typename std::remove_pointer<T>::type>(param.name);
     std::string cliName = (param.alias != '\0') ?
         "-" + std::string(1, param.alias) + ",--" + mappedName :
         "--" + mappedName;
   
     // Note that we have to add the option as type equal to the mapped type, not
     // the true type of the option.
     AddToCLI11<typename std::remove_pointer<T>::type>(
         cliName, param, *app);
   }
   
   } // namespace cli
   } // namespace bindings
   } // namespace mlpack
   
   #endif

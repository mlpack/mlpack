/**
 * @file print_input_param_impl.hpp
 * @author Vasyl Teliman
 *
 * Print Java code to handle input arguments.
 */
#ifndef MLPACK_BINDINGS_JAVA_PRINT_INPUT_PARAM_IMPL_HPP
#define MLPACK_BINDINGS_JAVA_PRINT_INPUT_PARAM_IMPL_HPP

#include "strip_type.hpp"
#include "get_java_type.hpp"

namespace mlpack {
namespace bindings {
namespace java {

/**
 * Print the input processing (basically calling CLI::GetParam<>()) for a
 * non-serializable type.
 */
template<typename T>
void PrintInputParam(
    const util::ParamData& d,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type*,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  if (d.name == "verbose")
  {
    std::cout << "    {" << std::endl
       << "      String name = \"verbose\";" << std::endl
       << "      Boolean value = params.get(name, Boolean.class);" << std::endl
       << "      if (value != null && value == true) {" << std::endl
       << "        CLI.enableVerbose();" << std::endl
       << "      } else {" << std::endl
       << "        CLI.disableVerbose();" << std::endl
       << "      }" << std::endl
       << "    }" << std::endl
       << std::endl;
    return;
  }

  std::string typeName;
  if (std::is_same<T, bool>::value)
    typeName = "Bool";
  else if (std::is_same<T, int>::value)
    typeName = "Int";
  else if (std::is_same<T, double>::value)
    typeName = "Double";
  else if (std::is_same<T, std::string>::value)
    typeName = "String";
  else if (std::is_same<T, std::vector<std::string>>::value)
    typeName = "VecString";
  else if (std::is_same<T, std::vector<int>>::value)
    typeName = "VecInt";
  else
    typeName = "Unknown";

  
  std::string javaClass;
  if (std::is_same<T, std::vector<std::string>>::value ||
      std::is_same<T, std::vector<int>>::value)
    javaClass = "List";
  else
    javaClass = GetJavaType<T>(d);

  if (d.required)
  {
    std::cout << "    {" << std::endl
              << "      String name = \"" << d.name << "\";" << std::endl
              << "      checkHasRequiredParameter(params, name);" << std::endl
              << "      " << javaClass << " value = params.get(name, " << javaClass << ".class);" << std::endl
              << "      CLI.set" << typeName << "Param(name, value);" << std::endl
              << "      CLI.setPassed(name);" << std::endl
              << "    }" << std::endl;
  }
  else
  {
    std::cout << "    {" << std::endl
              << "      String name = \"" << d.name << "\";" << std::endl
              << "      " << javaClass << " value = params.get(name, " << javaClass << ".class);" << std::endl
              << "      if (value != null) {" << std::endl
              << "        CLI.set" << typeName << "Param(name, value);" << std::endl
              << "        CLI.setPassed(name);" << std::endl
              << "      }" << std::endl
              << "    }" << std::endl;
  }
}

/**
 * Print the input processing for an Armadillo type.
 */
template<typename T>
void PrintInputParam(
    const util::ParamData& d,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type*,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  std::string uChar = (std::is_same<typename T::elem_type, size_t>::value) ?
      "U" : "";
  std::string matTypeSuffix = "";
  std::string extra = "";
  if (T::is_row)
  {
    matTypeSuffix = "Row";
  }
  else if (T::is_col)
  {
    matTypeSuffix = "Col";
  }
  else
  {
    matTypeSuffix = "Mat";
  }

  if (d.required)
  {
    std::cout << "    {" << std::endl
              << "      String name = \"" << d.name << "\";" << std::endl
              << "      checkHasRequiredParameter(params, name);" << std::endl
              << "      INDArray value = params.get(name, INDArray.class);" << std::endl
              << "      CLI.set" << uChar << matTypeSuffix << "Param(name, value);" << std::endl
              << "      CLI.setPassed(name);" << std::endl
              << "    }" << std::endl;
  }
  else
  {
    std::cout << "    {" << std::endl
              << "      String name = \"" << d.name << "\";" << std::endl
              << "      INDArray value = params.get(name, INDArray.class);" << std::endl
              << "      if (value != null) {" << std::endl
              << "        CLI.set" << uChar << matTypeSuffix << "Param(name, value);" << std::endl
              << "        CLI.setPassed(name);" << std::endl
              << "      }" << std::endl
              << "    }" << std::endl;
  }
}

/**
 * Print the input processing for a serializable type.
 */
template<typename T>
void PrintInputParam(
    const util::ParamData& d,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type*,
    const typename std::enable_if<data::HasSerialize<T>::value>::type*,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  std::string type = StripType(d.cppType);

  if (d.required)
  {
    std::cout << "    {" << std::endl
              << "      String name = \"" << d.name << "\";" << std::endl
              << "      checkHasRequiredParameter(params, name);" << std::endl
              << "      " << type << " value = params.get(name, " << type << ".class);" << std::endl
              << "      set" << type << "Ptr(name, value.getPointer());" << std::endl
              << "      CLI.setPassed(name);" << std::endl
              << "    }" << std::endl;
  }
  else
  {
    std::cout << "    {" << std::endl
              << "      String name = \"" << d.name << "\";" << std::endl
              << "      " << type << " value = params.get(name, " << type << ".class);" << std::endl
              << "      if (value != null) {" << std::endl
              << "        set" << type << "Ptr(name, value.getPointer());" << std::endl
              << "        CLI.setPassed(name);" << std::endl
              << "      }" << std::endl
              << "    }" << std::endl;
  }
}

/**
 * Print the input processing (basically calling CLI::GetParam<>()) for a
 * matrix with DatasetInfo type.
 */
template<typename T>
void PrintInputParam(
    const util::ParamData& d,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  if (d.required)
  {
    std::cout << "    {" << std::endl
              << "      String name = \"" << d.name << "\";" << std::endl
              << "      checkHasRequiredParameter(params, name);" << std::endl
              << "      MatrixWithInfo value = params.get(name, MatrixWithInfo.class);" << std::endl
              << "      CLI.setMatWithInfoParam(name, value);" << std::endl
              << "      CLI.setPassed(name);" << std::endl
              << "    }" << std::endl;
  }
  else
  {
    std::cout << "    {" << std::endl
              << "      String name = \"" << d.name << "\";" << std::endl
              << "      MatrixWithInfo value = params.get(name, MatrixWithInfo.class);" << std::endl
              << "      if (value != null) {" << std::endl
              << "        CLI.setMatWithInfoParam(name, value);" << std::endl
              << "        CLI.setPassed(name);" << std::endl
              << "      }" << std::endl
              << "    }" << std::endl;
  }
}

} // namespace java
} // namespace bindings
} // namespace mlpack

#endif

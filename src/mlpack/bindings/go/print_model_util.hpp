/**
 * @file bindings/go/print_model_util.hpp
 * @author Yasmine Dumouchel
 *
 * Print the functions and structs associated with serializable model.
 * for generating the .cpp, .h, and .go binding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_PRINT_CLASS_DEFN_HPP
#define MLPACK_BINDINGS_GO_PRINT_CLASS_DEFN_HPP

#include "strip_type.hpp"

namespace mlpack {
namespace bindings {
namespace go {

/**
 * Non-serializable models don't require any special definitions, so this prints
 * nothing.
 */
template<typename T>
void PrintModelUtilCPP(
    const util::ParamData& /* d */,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  // Do nothing.
}

/**
 * Matrices don't require any special definitions, so this prints nothing.
 */
template<typename T>
void PrintModelUtilCPP(
    const util::ParamData& /* d */,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
{
  // Do nothing.
}

/**
 * Matrices with Info don't require any special definitions, so this prints nothing.
 */
template<typename T>
void PrintModelUtilCPP(
    const util::ParamData& /* d */,
    const typename boost::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  // Do nothing.
}

/**
 * Serializable models require a special class definition.
 */
template<typename T>
void PrintModelUtilCPP(
    const util::ParamData& d,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  // First, we have to parse the type.  If we have something like, e.g.,
  // 'LogisticRegression<>', we must convert this to 'LogisticRegression[].'
  std::string goStrippedType, strippedType, printedType, defaultsType;
  StripType(d.cppType, goStrippedType, strippedType, printedType, defaultsType);
  /**
   * This gives us code like:
   *
   *  extern "C" void mlpackSet\<Type\>Ptr(
   *                 const char* identifier,
   *                 void *value)
   *  {
   *    SetParamPtr\<Type\>(identifier,
   *                      static_cast\<Type\>*(value));
   *  }
   *
   */
  std::cout << "extern \"C\" void mlpackSet" << strippedType
            << "Ptr("  << std::endl;
  std::cout << "    const char* identifier, " << std::endl;
  std::cout << "    void* value)" << std::endl;
  std::cout << "{" << std::endl;
  std::cout << "  SetParamPtr<" << printedType
            << ">(identifier," << std::endl;
  std::cout << "      static_cast<" << printedType
            << "*>(value));" << std::endl;
  std::cout << "}" << std::endl;
  std::cout << std::endl;

  /**
   * This gives us code like:
   *
   *  extern "C" void *mlpackGet\<Type\>Ptr(const char* identifier)
   *  {
   *    \<Type\> *modelptr = GetParamPtr\<Type\>(identifier);
   *    return modelptr;
   *  }
   *
   */
  std::cout << "extern \"C\" void *mlpackGet" << strippedType
            << "Ptr(const char* identifier)" << std::endl;
  std::cout << "{" << std::endl;
  std::cout << "  " << printedType << " *modelptr = GetParamPtr<"
            << printedType << ">(identifier);" << std::endl;
  std::cout << "  return modelptr;" << std::endl;
  std::cout << "}" << std::endl;
  std::cout << std::endl;
}

/**
 * Print the function to set and get serialization models from Go to mlpack.
 *
 * @param d Parameter data.
 * @param * (input) Unused parameter.
 * @param * (output) Unused parameter.
 */
template<typename T>
void PrintModelUtilCPP(const util::ParamData& d,
                       const void* /* input */,
                       void* /* output */)
{
  PrintModelUtilCPP<typename std::remove_pointer<T>::type>(d);
}


/**
 * Non-serializable models don't require any special definitions, so this prints
 * nothing.
 */
template<typename T>
void PrintModelUtilH(
    const util::ParamData& /* d */,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)

{
  // Do nothing.
}

/**
 * Matrices don't require any special definitions, so this prints nothing.
 */
template<typename T>
void PrintModelUtilH(
    const util::ParamData& /* d */,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
{
  // Do nothing.
}

/**
 * Matrices with Info don't require any special definitions, so this prints nothing.
 */
template<typename T>
void PrintModelUtilH(
    const util::ParamData& /* d */,
    const typename boost::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  // Do nothing.
}

/**
 * Serializable models require a special class definition.
 */
template<typename T>
void PrintModelUtilH(
    const util::ParamData& d,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  // First, we have to parse the type.  If we have something like, e.g.,
  // 'LogisticRegression<>', we must convert this to 'LogisticRegression[].'
  std::string goStrippedType, strippedType, printedType, defaultsType;
  StripType(d.cppType, goStrippedType, strippedType, printedType, defaultsType);

  /**
   * This gives us code like:
   *
   *  extern void *mlpackSet\<Type\>Ptr(const char* identifier, void* value);
   *
   */
  std::cout << "extern void mlpackSet" << strippedType
            << "Ptr(const char* identifier, void* value);" << std::endl;
  std::cout << std::endl;

  /**
   * This gives us code like:
   *
   *  extern void *mlpackGet\<Type\>Ptr(const char* identifier);
   *
   */
  std::cout << "extern void *mlpackGet" << strippedType
            << "Ptr(const char* identifier);" << std::endl;
  std::cout << std::endl;
}

/**
 * Print the function to set and get serialization models from Go to mlpack.
 *
 * @param d Parameter data.
 * @param * (input) Unused parameter.
 * @param * (output) Unused parameter.
 */
template<typename T>
void PrintModelUtilH(const util::ParamData& d,
                     const void* /* input */,
                     void* /* output */)
{
  PrintModelUtilH<typename std::remove_pointer<T>::type>(d);
}

/**
 * Non-serializable models don't require any special definitions, so this prints
 * nothing.
 */
template<typename T>
void PrintModelUtilGo(
    const util::ParamData& /* d */,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  // Do nothing.
}

/**
 * Matrices don't require any special definitions, so this prints nothing.
 */
template<typename T>
void PrintModelUtilGo(
    const util::ParamData& /* d */,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
{
  // Do nothing.
}

/**
 * Matrices with Info don't require any special definitions, so this prints nothing.
 */
template<typename T>
void PrintModelUtilGo(
    const util::ParamData& /* d */,
    const typename boost::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  // Do nothing.
}

/**
 * Serializable models require a special class definition.
 */
template<typename T>
void PrintModelUtilGo(
    const util::ParamData& /* d */,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  // Do nothing.
}

/**
 * Print the Go struct for Go serialization model and their associated
 * set and get methods.
 *
 * @param d Parameter data.
 * @param * (input) Unused parameter.
 * @param * (output) Unused parameter.
 */
template<typename T>
void PrintModelUtilGo(const util::ParamData& d,
                      const void* /* input */,
                      void* /* output */)
{
  PrintModelUtilGo<typename std::remove_pointer<T>::type>(d);
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif

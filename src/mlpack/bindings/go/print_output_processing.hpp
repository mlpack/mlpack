/**
 * @file bindings/go/print_output_processing.hpp
 * @author Yasmine Dumouchel
 *
 * Print the output processing in a Go binding .go file for a given
 * parameter.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_PRINT_OUTPUT_PROCESSING_HPP
#define MLPACK_BINDINGS_GO_PRINT_OUTPUT_PROCESSING_HPP

#include <mlpack/prereqs.hpp>
#include "get_type.hpp"
#include "strip_type.hpp"
#include <mlpack/bindings/util/camel_case.hpp>

namespace mlpack {
namespace bindings {
namespace go {

/**
 * Print output processing for a regular parameter type.
 */
template<typename T>
void PrintOutputProcessing(
    util::ParamData& d,
    const size_t indent,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  const std::string prefix(indent, ' ');

  /**
   * This gives us code like:
   *
   *  \<paramName\> := GetParam\<Type\>("paramName")
   *
   */

  std::string name = d.name;
  name = util::CamelCase(name, true);
  std::cout << prefix << name << " := getParam" << GetType<T>(d)
            << "(\"" << d.name << "\")" << std::endl;
}

/**
 * Print output processing for a matrix type.
 */
template<typename T>
void PrintOutputProcessing(
    util::ParamData& d,
    const size_t indent,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
{
  const std::string prefix(indent, ' ');

  /**
   * This gives us code like:
   *
   *  var \<paramName\>Ptr mlpackArma
   *  \<paramName\> := \<paramName\>_ptr.ArmaToGonum_\<Type\>("paramName")
   *
   */
  std::string name = d.name;
  name = util::CamelCase(name, true);
  std::cout << prefix << "var " << name << "Ptr mlpackArma" << std::endl;
  std::cout << prefix << name << " := " << name
            << "Ptr.armaToGonum" << GetType<T>(d)
            << "(\""  << d.name << "\")" << std::endl;
}
/**
 * Print output processing for a matrix with info type.
 */
template<typename T>
void PrintOutputProcessing(
    util::ParamData& d,
    const size_t indent,
    const typename boost::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  const std::string prefix(indent, ' ');

  /**
   * This gives us code like:
   *
   *  var \<paramName\>_ptr mlpackArma
   *  \<paramName\> := \<paramName\>Ptr.ArmaToGonumWithInfo\<Type\>("paramName")
   *
   */
  std::string name = d.name;
  name = util::CamelCase(name, true);
  std::cout << prefix << "var " << name << "Ptr mlpackArma" << std::endl;
  std::cout << prefix << name << " := " << name << "Ptr.armaToGonumWith"
            << "Info(\""  << d.name << "\")" << std::endl;
}

/**
 * Print output processing for a serializable model.
 */
template<typename T>
void PrintOutputProcessing(
    util::ParamData& d,
    const size_t indent,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  // Get the type names we need to use.
  std::string goStrippedType, strippedType, printedType, defaultsType;
  StripType(d.cppType, goStrippedType, strippedType, printedType, defaultsType);

  const std::string prefix(indent, ' ');

  /**
   * This gives us code like:
   *
   *  var modelOut \<Type\>
   *  modelOut.get\<Type\>("paramName")
   *
   */
  std::string name = d.name;
  name = util::CamelCase(name, true);
  std::cout << prefix << "var " << name << " " << goStrippedType << std::endl;
  std::cout << prefix << name << ".get" << strippedType
            << "(\"" << d.name << "\")" << std::endl;
}

/**
 * @param d Parameter data struct.
 * @param * (input) Unused parameter.
 * @param * (output) Unused parameter.
 */
template<typename T>
void PrintOutputProcessing(util::ParamData& d,
                           const void* /*input*/,
                           void* /* output */)
{
  PrintOutputProcessing<typename std::remove_pointer<T>::type>(d, 2);
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif

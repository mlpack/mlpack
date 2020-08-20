/**
 * @file bindings/markdown/program_doc_wrapper.hpp
 * @author Ryan Curtin
 *
 * A simple wrapper around programName, shortDescription, longDescription,
 * example and seeAlso that also respectively register all the macros upon
 * construction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_MARKDOWN_PROGRAM_DOC_WRAPPER_HPP
#define MLPACK_BINDINGS_MARKDOWN_PROGRAM_DOC_WRAPPER_HPP

#include "binding_info.hpp"

namespace mlpack {
namespace bindings {
namespace markdown {

class ProgramNameWrapper
{
 public:
  /**
   * Register programName.
   */
  ProgramNameWrapper(const std::string& bindingName,
                     const std::string& programName)
  {
    BindingInfo::GetSingleton().map[bindingName].programName =
        std::move(programName);
  }
};

class ShortDescriptionWrapper
{
 public:
  /**
   * Register shortDescription.
   */
  ShortDescriptionWrapper(const std::string& bindingName,
                          const std::string& shortDescription)
  {
    BindingInfo::GetSingleton().map[bindingName].shortDescription =
        std::move(shortDescription);
  }
};

class LongDescriptionWrapper
{
 public:
  /**
   * Register longDescription.
   */
  LongDescriptionWrapper(const std::string& bindingName,
                         const std::function<std::string()>& longDescription)
  {
    BindingInfo::GetSingleton().map[bindingName].longDescription =
        std::move(longDescription);
  }
};

class ExampleWrapper
{
 public:
  /**
   * Register example.
   */
  ExampleWrapper(const std::string& bindingName,
                 const std::function<std::string()>& example)
  {
    BindingInfo::GetSingleton().map[bindingName].example.push_back(
        std::move(example));
  }
};

class SeeAlsoWrapper
{
 public:
  /**
   * Register seeAlso.
   */
  SeeAlsoWrapper(const std::string& bindingName,
                 const std::string& description, const std::string& link)
  {
    BindingInfo::GetSingleton().map[bindingName].seeAlso.push_back(
        std::move(std::make_pair(description, link)));
  }
};

} // namespace markdown
} // namespace bindings
} // namespace mlpack

#endif

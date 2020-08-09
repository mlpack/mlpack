/**
 * @file bindings/markdown/program_doc_wrapper.hpp
 * @author Ryan Curtin
 *
 * A simple wrapper around ProgramName, ShortDescription, LongDescription,
 * Example and SeeAlso that also respectively calls 
 * BindingInfo::RegisterProgramName(), BindingInfo::RegisterShortDescription(),
 * BindingInfo::RegisterLongDescription(), BindingInfo::RegisterExample() and 
 * BindingInfo::RegisterSeeAlso() upon construction.
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
   * Construct a ProgramName object and register it with
   * BindingInfo::RegisterProgramName().
   */
  ProgramNameWrapper(const std::string& bindingName,
                     const std::string& programName)
  {
    util::ProgramName pd(programName);
    BindingInfo::RegisterProgramName(bindingName, pd);
  }
};

class ShortDescriptionWrapper
{
 public:
  /**
   * Construct a ShortDescription object and register it with
   * BindingInfo::RegisterShortDescription().
   */
  ShortDescriptionWrapper(const std::string& bindingName,
                          const std::string shortDescription)
  {
    util::ShortDescription pd(shortDescription);
    BindingInfo::RegisterShortDescription(bindingName, pd);
  }
};

class LongDescriptionWrapper
{
 public:
  /**
   * Construct a LongDescription object and register it with
   * BindingInfo::RegisterLongDescription().
   */
  LongDescriptionWrapper(const std::string& bindingName,
                         const std::function<std::string()> longDescription)
  {
    util::LongDescription pd(longDescription);
    BindingInfo::RegisterLongDescription(bindingName, pd);
  }
};

class ExampleWrapper
{
 public:
  /**
   * Construct a Example object and register it with
   * BindingInfo::RegisterExample().
   */
  ExampleWrapper(const std::string& bindingName,
                 const std::function<std::string()> example)
  {
    util::Example pd(example);
    BindingInfo::RegisterExample(bindingName, pd);
  }
};

class SeeAlsoWrapper
{
 public:
  /**
   * Construct a SeeAlso object and register it with
   * BindingInfo::RegisterSeeAlso().
   */
  SeeAlsoWrapper(const std::string& bindingName,
                 const std::string description, const std::string link)
  {
    util::SeeAlso pd(description, link);
    BindingInfo::RegisterSeeAlso(bindingName, pd);
  }
};

} // namespace markdown
} // namespace bindings
} // namespace mlpack

#endif

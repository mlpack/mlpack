/**
 * @file bindings/markdown/binding_info.hpp
 * @author Ryan Curtin
 *
 * This file defines the BindingInfo singleton class that is used specifically
 * for the Markdown bindings to map from a binding name (i.e. "knn") to
 * multiple documentation objects, which are then used to generate the
 * documentation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_MARKDOWN_BINDING_NAME_HPP
#define MLPACK_BINDINGS_MARKDOWN_BINDING_NAME_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/program_doc.hpp>

namespace mlpack {
namespace bindings {
namespace markdown {

/**
 * The BindingInfo class is used by the Markdown documentation generator to
 * store multiple documentation objects, indexed by both the binding name (i.e.
 * "knn") and the language (i.e. "cli").
 */
class BindingInfo
{
 public:
  //! Return a ProgramName object for a given bindingName.
  static util::ProgramName& GetProgramName(const std::string& bindingName);

  //! Return a ShortDescription object for a given bindingName.
  static util::ShortDescription& GetShortDescription(
      const std::string& bindingName);

  //! Return a LongDescription object for a given bindingName.
  static util::LongDescription& GetLongDescription(
      const std::string& bindingName);

  //! Return a Example object for a given bindingName.
  static std::vector<util::Example>& GetExample(const std::string& bindingName);

  //! Return a SeeAlso object for a given bindingName.
  static std::vector<util::SeeAlso>& GetSeeAlso(const std::string& bindingName);

  //! Register a ProgramName object with the given bindingName.
  static void RegisterProgramName(const std::string& bindingName,
                                  const util::ProgramName& programName);

  //! Register a ShortDescription object with the given bindingName.
  static void RegisterShortDescription(const std::string& bindingName,
                                       const util::ShortDescription&
                                             shortDescription);

  //! Register a LongDescription object with the given bindingName.
  static void RegisterLongDescription(const std::string& bindingName,
                                      const util::LongDescription&
                                            longDescription);

  //! Register a Example object with the given bindingName.
  static void RegisterExample(const std::string& bindingName,
                              const util::Example& example);

  //! Register a SeeAlso object with the given bindingName.
  static void RegisterSeeAlso(const std::string& bindingName,
                              const util::SeeAlso& seeAlso);

  //! Get or modify the current language (don't set it to something invalid!).
  static std::string& Language();

 private:
  //! Private constructor, so that only one instance can be created.
  BindingInfo() { }

  //! Get the singleton.
  static BindingInfo& GetSingleton();

  //! Internally-held map for mapping a binding name to a ProgramName name.
  std::unordered_map<std::string, util::ProgramName> mapProgramName;

  //! Internally-held map for mapping a binding name to a ShortDescription name.
  std::unordered_map<std::string, util::ShortDescription> mapShortDescription;

  //! Internally-held map for mapping a binding name to a LongDescription name.
  std::unordered_map<std::string, util::LongDescription> mapLongDescription;

  //! Internally-held map for mapping a binding name to a Example name.
  std::unordered_map<std::string, std::vector<util::Example>> mapExample;

  //! Internally-held map for mapping a binding name to a SeeAlso name.
  std::unordered_map<std::string, std::vector<util::SeeAlso>> mapSeeAlso;

  //! Holds the name of the language that we are currently printing.  This is
  //! modified before printing the documentation, and then used by
  //! print_doc_functions.hpp during printing to print the correct language.
  std::string language;
};

} // namespace markdown
} // namespace bindings
} // namespace mlpack

#endif

/**
 * @file program_doc_wrapper.hpp
 * @author Ryan Curtin
 *
 * A simple wrapper around ProgramDoc that also calls
 * BindingInfo::RegisterProgramDoc() upon construction.
 */
#ifndef MLPACK_BINDINGS_MARKDOWN_PROGRAM_DOC_WRAPPER_HPP
#define MLPACK_BINDINGS_MARKDOWN_PROGRAM_DOC_WRAPPER_HPP

namespace mlpack {
namespace bindings {
namespace markdown {

class ProgramDocWrapper
{
 public:
  /**
   * Construct a ProgramDoc object and register it with
   * BindingInfo::RegisterProgramDoc().
   */
  ProgramDocWrapper(const std::string& bindingName,
                    const std::string& programName,
                    const std::function<std::string()>& documentation)
  {
    util::ProgramDoc pd(programName, documentation);
    BindingInfo::RegisterProgramDoc(bindingName, pd);
  }
};

} // namespace markdown
} // namespace bindings
} // namespace mlpack

#endif

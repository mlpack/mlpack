/**
 * @file print_h.cpp
 * @author Yasmine Dumouchel
 *
 * Implementation of function to generate a .h file given a list of parameters
 * for the function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "print_h.hpp"
#include "camel_case.hpp"
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/hyphenate_string.hpp>
#include <set>

using namespace mlpack::util;
using namespace std;

namespace mlpack {
namespace bindings {
namespace go {

/**
 * Given a list of parameter definition and program documentation, print a
 * generated .h file to stdout.
 *
 * @param parameters List of parameters the program will use (from CLI).
 * @param programInfo Documentation for the program.
 * @param functionName Name of the function (i.e. "pca").
 */
void PrintH(const util::ProgramDoc& programInfo,
            const std::string& functionName)
{
  // Restore parameters.
  CLI::RestoreSettings(programInfo.programName);

  const std::map<std::string, util::ParamData>& parameters = CLI::Parameters();
  typedef std::map<std::string, util::ParamData>::const_iterator ParamIter;

  // First, we must generate the header comment and namespace.
  cout << "#include <stdint.h>" << endl;
  cout << "#include <stddef.h>" << endl;
  cout << endl;
  cout << "#if defined(__cplusplus) || defined(c_plusplus)" << endl;
  cout << "extern \"C\" {" << endl;
  cout << "#endif" << endl;
  cout << endl;

  // Then we must print utility function for model type parameters if needed.
  for (ParamIter it = parameters.begin(); it != parameters.end(); ++it)
  {
    const util::ParamData& d = it->second;
    if (d.input)
      CLI::GetSingleton().functionMap[d.tname]["PrintModelUtilH"](d,
                                                NULL, NULL);
  }

  std::string goFunctionName = CamelCase(functionName);
  // We generate the wrapper function for mlpackMain().
  cout << "extern void mlpack" << goFunctionName << "();" << endl;
  cout << endl;

  // Finally we close print the closing bracket for extern C.
  cout << "#if defined(__cplusplus) || defined(c_plusplus)" << endl;
  cout << "}" << endl;
  cout << "#endif" << endl;
}

} // namespace go
} // namespace bindings
} // namespace mlpack

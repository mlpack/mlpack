/**
 * @file print_cpp.cpp
 * @author Yasmine Dumouchel
 *
 * Implementation of function to generate a .cpp file given a list of parameters
 * for the function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "print_cpp.hpp"
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
 * generated .cpp file to stdout.
 *
 * @param parameters List of parameters the program will use (from CLI).
 * @param programInfo Documentation for the program.
 * @param functionName Name of the function (i.e. "pca").
 */
void PrintCPP(const ProgramDoc& programInfo,
              const string& mainFilename,
              const string& functionName)
{
  // Restore parameters.
  CLI::RestoreSettings(programInfo.programName);

  const std::map<std::string, util::ParamData>& parameters = CLI::Parameters();
  typedef std::map<std::string, util::ParamData>::const_iterator ParamIter;

  // First, we must generate the header comment and namespace.
  cout << "#include \"" << functionName << ".h\""<< endl;
  cout << "#include <" << mainFilename << ">" << endl;
  cout << "#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>" << endl;
  cout << endl;
  cout << "using namespace mlpack;" << endl;
  cout << "using namespace mlpack::util;" << endl;
  cout << "using namespace std;" << endl;
  cout << endl;

  // Then we must print utility function for model type parameters if needed.
  for (ParamIter it = parameters.begin(); it != parameters.end(); ++it)
  {
    const util::ParamData& d = it->second;
    if (d.input)
      CLI::GetSingleton().functionMap[d.tname]["PrintModelUtilCPP"](d,
                                                 NULL, NULL);
  }

  // Finally, we generate the wrapper function for mlpackMain().
  std::string goFunctionName = CamelCase(functionName);
  cout << "static void " << goFunctionName << "MlpackMain()" << endl;
  cout << "{" << endl;
  cout << "  " << "mlpackMain();" << endl;
  cout << "}" << endl;
  cout << endl;
  cout << "extern \"C\" void mlpack" << goFunctionName << "()" << endl;
  cout << "{" << endl;
  cout << "  " << goFunctionName << "MlpackMain();" << endl;
  cout << "}" << endl;
  cout << endl;
}

} // namespace go
} // namespace bindings
} // namespace mlpack

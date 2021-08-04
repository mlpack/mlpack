#ifndef MLPACK_BINDINGS_MARKDOWN_PRINT_PARAM_TABLE_HPP
#define MLPACK_BINDINGS_MARKDOWN_PRINT_PARAM_TABLE_HPP

#include <mlpack/prereqs.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::util;
using namespace mlpack::bindings;
using namespace mlpack::bindings::markdown;

void PrintParamTable(const string& bindingName,
                     const string& language,
                     Params& params,
                     const set<string>& headers,
                     unordered_set<string>& paramsSet,
                     const bool onlyHyperParams,
                     const bool onlyMatrixParams,
                     const bool onlyInputParams,
                     const bool onlyOutputParams);

#endif

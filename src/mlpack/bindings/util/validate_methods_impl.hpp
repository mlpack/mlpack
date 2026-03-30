/**
 * @file validate_methods_impl.hpp
 * @author Ryan Curtin
 *
 * Ensure that the parameters for each of the individual methods of a grouped
 * binding satisfy a handful of requirements.
 */
#ifndef MLPACK_BINDINGS_UTIL_VALIDATE_METHODS_IMPL_HPP
#define MLPACK_BINDINGS_UTIL_VALIDATE_METHODS_IMPL_HPP

#include "validate_methods.hpp"

namespace mlpack {
namespace bindings {
namespace util {

/**
 * Check the validity of each of the methods of a grouped binding:
 *
 *  - The "train" binding should have one or two required parameters of matrix
 *    type, and output only one model parameter.
 *
 *  - A "predict"/"classify"/"probabilities" binding should have two required
 *    parameters, one of matrix type and one of model type, and output only one
 *    matrix parameter.
 *
 *  - All methods should have only one output parameter.
 */
inline void ValidateMethods(
    const std::vector<std::string>& methods,
    const std::map<std::string, mlpack::util::Params>& params,
    const std::string& callerName)
{
  for (const std::string& m : methods)
  {
    std::map<std::string, mlpack::util::ParamData*> reqInputParams;
    std::map<std::string, mlpack::util::ParamData*> outputParams;

    mlpack::util::Params& ps = const_cast<mlpack::util::Params&>(params.at(m));
    for (auto& it : ps.Parameters())
    {
      mlpack::util::ParamData& p = it.second;

      if (p.required && p.input)
        reqInputParams[p.name] = &p;
      else if (!p.input)
        outputParams[p.name] = &p;
    }

    if (m == "train")
    {
      // We should have one or two required parameters of matrix type.
      if (reqInputParams.size() == 0)
      {
        std::ostringstream oss;
        oss << callerName << ": training binding '" << m << "' has no required "
            << "input parameters, but training bindings must have one or two "
            << "required input parameters!";
        throw std::runtime_error(oss.str());
      }
      else if (reqInputParams.size() > 2)
      {
        std::ostringstream oss;
        oss << callerName << ": training binding '" << m << "' has "
            << reqInputParams.size() << " required input parameters, but "
            << "training bindings must have one or two required input "
            << "parameters!";
        throw std::runtime_error(oss.str());
      }
    }
    else if (m == "predict" || m == "classify" || m == "probabilities")
    {
      // We should have two required input parameters.
      if (reqInputParams.size() != 2)
      {
        std::ostringstream oss;
        oss << callerName << ": prediction binding '" << m << "' has "
            << reqInputParams.size() << " required input parameters, but "
            << "prediction bindings must have only two required input "
            << "parameter!";
        throw std::runtime_error(oss.str());
      }
    }

    // Check the type of the input parameters; it must be arma::mat or
    // std::tuple<mlpack::DatasetInfo, arma::mat>.
    for (auto& it : reqInputParams)
    {
      bool allowSerializable = !(m == "train");
      bool serializable;
      ps.functionMap[it.second->tname]["IsSerializable"](*(it.second), NULL,
          (void*) &serializable);

      if (it.second->cppType != "arma::mat" &&
          it.second->cppType != "arma::rowvec" &&
          it.second->cppType != "arma::Row<size_t>" &&
          it.second->cppType != "std::tuple<mlpack::DatasetInfo, arma::mat>" &&
          !(allowSerializable && serializable))
      {
        std::ostringstream oss;
        oss << callerName << ": binding required input parameter '" << it.first
            << "' has type '" << it.second->cppType << "', but required input "
            << "parameters must have type arma::mat or "
            << "std::tuple<mlpack::DatasetInfo, arma::mat>!";
        throw std::runtime_error(oss.str());
      }
    }

    // The binding should have only one output parameter.
    if (outputParams.size() != 1)
    {
      std::ostringstream oss;
      oss << callerName << ": binding '" << m << "' has " << outputParams.size()
          << " output parameters, but must have only one!";
      throw std::runtime_error(oss.str());
    }

    if (m == "train")
    {
      // The training binding should have a serializable model type output
      // parameter.
      mlpack::util::ParamData* p = outputParams.begin()->second;
      bool serializable;
      ps.functionMap[p->tname]["IsSerializable"](*p, NULL,
          (void*) &serializable);

      if (!serializable)
      {
        std::ostringstream oss;
        oss << callerName << ": training binding output parameter '" << p->name
            << "' has type '" << p->cppType << "' but must have a serializable "
            << "type!";
        throw std::runtime_error(oss.str());
      }
    }
    else if (m == "predict" || m == "classify" || m == "probabilities")
    {
      // Prediction bindings should have a matrix type output parameter.
      mlpack::util::ParamData* p = outputParams.begin()->second;
      if (p->cppType != "arma::mat" &&
          p->cppType != "std::tuple<mlpack::DatasetInfo, arma::mat>")
      {
        std::ostringstream oss;
        oss << callerName << ": binding required input parameter '" << p->name
            << "' has type '" << p->cppType << "', but required input "
            << "parameters must have type arma::mat or "
            << "std::tuple<mlpack::DatasetInfo, arma::mat>!";
        throw std::runtime_error(oss.str());
      }
    }
  }
}

} // namespace util
} // namespace bindings
} // namespace mlpack

#include "validate_methods_impl.hpp"

#endif

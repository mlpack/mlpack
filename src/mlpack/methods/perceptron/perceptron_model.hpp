/**
 * @file methods/perceptron/perceptron_model.hpp
 * @author Udit Saxena
 * @author Dirk Eddelbuettel
 *
 * A serializable perceptron, used by the perceptron binding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_PERCEPTRON_PERCEPTRON_MODEL_HPP
#define MLPACK_METHODS_PERCEPTRON_PERCEPTRON_MODEL_HPP

#include <mlpack/core.hpp>

namespace mlpack {

// When we save a model, we must also save the class mappings.  So we use this
// auxiliary structure to store both the perceptron and the mapping, and we'll
// save this.
class PerceptronModel
{
 private:
  Perceptron<> p;
  arma::Col<size_t> map;

 public:
  Perceptron<>& P() { return p; }
  const Perceptron<>& P() const { return p; }

  arma::Col<size_t>& Map() { return map; }
  const arma::Col<size_t>& Map() const { return map; }

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(p));
    ar(CEREAL_NVP(map));
  }
};

} // namespace mlpack

#endif

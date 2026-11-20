/**
 * @file core/cereal/low_precision.hpp
 * @author Ryan Curtin
 *
 * Extra shims necessary for cereal to serialize to JSON for low-precision types
 * (e.g.  FP16, BF16, etc.).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CEREAL_LOW_PRECISION_HPP
#define MLPACK_CORE_CEREAL_LOW_PRECISION_HPP

namespace cereal {

// Because our serialization is always done with name-value pairs, we can catch
// any FP16 serialization at the NVP level with a specialized implementation of
// the load and save functions for the JSON archive (the only one that does not
// serialize low-precision correctly).

#if defined(ARMA_HAVE_FP16)

inline void CEREAL_SAVE_FUNCTION_NAME(JSONOutputArchive &ar,
                                      NameValuePair<arma::fp16&> const& t)
{
  ar.setNextName(t.name);
  std::ostringstream oss;
  oss.precision(std::numeric_limits<arma::fp16>::max_digits10);
  oss << t.value;
  ar(oss.str());
}

inline void CEREAL_LOAD_FUNCTION_NAME(JSONInputArchive& ar,
                                      NameValuePair<arma::fp16&>& t)
{
  ar.setNextName(t.name);
  std::string encoded;
  ar.loadValue(encoded);
  t.value = arma::fp16(std::stof(encoded));
}

#endif

} // namespace cereal

#endif

/**
 * @file core/cereal/low_precision.hpp
 * @author Omar Shrit
 *
 * Extra shims necessary for cereal to serialize low-precision types (e.g.
 * FP16, BF16, etc.).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CEREAL_LOW_PRECISION_HPP
#define MLPACK_CORE_CEREAL_LOW_PRECISION_HPP

#if defined(ARMA_HAVE_FP16)

namespace cereal {

inline void CEREAL_SAVE_FUNCTION_NAME(JSONOutputArchive& ar, arma::fp16 t)
{
  // Serialize FP16 to string in order to save.
  std::ostringstream oss;
  oss.precision(std::numeric_limits<arma::fp16>::max_digits10);
  oss << t;
  ar.saveValue(oss.str());
}

inline void CEREAL_LOAD_FUNCTION_NAME(JSONInputArchive& ar, arma::fp16& t)
{
  // Deserialize from string.
  std::string encoded;
  ar.loadValue(encoded);
  // Load as fp32, then convert to fp16.
  t = arma::fp16(std::stof(encoded));
}

} // namespace cereal

#endif

#endif

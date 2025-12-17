/**
 * @file core/data/load_model.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 *
 * Intenal implementation of model-specific Load() function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_MODEL_HPP
#define MLPACK_CORE_DATA_LOAD_MODEL_HPP

#include <cereal/archives/xml.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>

#include "text_options.hpp"

namespace mlpack {
namespace data {

template<typename Object>
bool LoadModel(Object& objectToSerialize,
               DataOptionsBase<PlainDataOptions>& opts,
               std::fstream& stream)
{
  try
  {
    if (opts.Format() == FileType::XML)
    {
      cereal::XMLInputArchive ar(stream);
      ar(cereal::make_nvp("model", objectToSerialize));
    }
    else if (opts.Format() == FileType::JSON)
    {
     cereal::JSONInputArchive ar(stream);
     ar(cereal::make_nvp("model", objectToSerialize));
    }
    else if (opts.Format() == FileType::BIN)
    {
      cereal::BinaryInputArchive ar(stream);
      ar(cereal::make_nvp("model", objectToSerialize));
    }

    return true;
  }
  catch (cereal::Exception& e)
  {
    if (opts.Fatal())
      Log::Fatal << e.what() << std::endl;
    else
      Log::Warn << e.what() << std::endl;

    return false;
  }
}

} // namespace data
} // namespace mlpack

#endif

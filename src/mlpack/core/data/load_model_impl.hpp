/**
 * @file core/data/load_model_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of model-specific Load() function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_MODEL_IMPL_HPP
#define MLPACK_CORE_DATA_LOAD_MODEL_IMPL_HPP

// In case it hasn't already been included.
#include "load.hpp"

#include "extension.hpp"

#include <cereal/archives/xml.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>

namespace mlpack {
namespace data {

// Load a model from file.
template<typename T>
bool Load(const std::string& filename,
          const std::string& name,
          T& t,
          const bool fatal,
          FileType f,
          std::enable_if_t<HasSerialize<T>::value>*)
{
  if (f == FileType::AutoDetect)
  {
    std::string extension = Extension(filename);

    if (extension == "xml")
      f = FileType::XML;
    else if (extension == "bin")
      f = FileType::BIN;
    else if (extension == "json")
      f = FileType::JSON;
    else
    {
      if (fatal)
        Log::Fatal << "Unable to detect type of '" << filename << "'; incorrect"
            << " extension?" << std::endl;
      else
        Log::Warn << "Unable to detect type of '" << filename << "'; load "
            << "failed.  Incorrect extension?" << std::endl;

      return false;
    }
  }

  // Now load the given format.
  std::ifstream ifs;
#ifdef _WIN32 // Open non-text in binary mode on Windows.
  if (f == FileType::BIN)
    ifs.open(filename, std::ifstream::in | std::ifstream::binary);
  else
    ifs.open(filename, std::ifstream::in);
#else
  ifs.open(filename, std::ifstream::in);
#endif

  if (!ifs.is_open())
  {
    if (fatal)
      Log::Fatal << "Unable to open file '" << filename << "' to load object '"
          << name << "'." << std::endl;
    else
      Log::Warn << "Unable to open file '" << filename << "' to load object '"
          << name << "'." << std::endl;

    return false;
  }
  try
  {
    if (f == FileType::XML)
    {
      cereal::XMLInputArchive ar(ifs);
      ar(cereal::make_nvp(name.c_str(), t));
    }
    else if (f == FileType::JSON)
    {
     cereal::JSONInputArchive ar(ifs);
     ar(cereal::make_nvp(name.c_str(), t));
    }
    else if (f == FileType::BIN)
    {
      cereal::BinaryInputArchive ar(ifs);
      ar(cereal::make_nvp(name.c_str(), t));
    }

    return true;
  }
  catch (cereal::Exception& e)
  {
    if (fatal)
      Log::Fatal << e.what() << std::endl;
    else
      Log::Warn << e.what() << std::endl;

    return false;
  }
}

} // namespace data
} // namespace mlpack

#endif

/**
 * @file load_model_impl.hpp
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

#include <algorithm>
#include <mlpack/core/util/timers.hpp>

#include "extension.hpp"

#include <boost/serialization/serialization.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

#include "serialization_shim.hpp"

namespace mlpack {
namespace data {

// Load a model from file.
template<typename T>
bool Load(const std::string& filename,
          const std::string& name,
          T& t,
          const bool fatal,
          format f)
{
  if (f == format::autodetect)
  {
    std::string extension = Extension(filename);

    if (extension == "xml")
      f = format::xml;
    else if (extension == "bin")
      f = format::binary;
    else if (extension == "txt")
      f = format::text;
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
  if (f == format::binary)
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
    if (f == format::xml)
    {
      boost::archive::xml_iarchive ar(ifs);
      ar >> CreateNVP(t, name);
    }
    else if (f == format::text)
    {
      boost::archive::text_iarchive ar(ifs);
      ar >> CreateNVP(t, name);
    }
    else if (f == format::binary)
    {
      boost::archive::binary_iarchive ar(ifs);
      ar >> CreateNVP(t, name);
    }

    return true;
  }
  catch (boost::archive::archive_exception& e)
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

/**
 * @file core/data/save_impl.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 *
 * Implementation of save functionality.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_SAVE_IMPL_HPP
#define MLPACK_CORE_DATA_SAVE_IMPL_HPP

// In case it hasn't already been included.
#include "save.hpp"
#include "extension.hpp"

namespace mlpack {
namespace data {

template<typename eT>
bool Save(const std::string& filename,
          const arma::Col<eT>& vec,
          const bool fatal,
          FileType inputSaveType)
{
  // Don't transpose: one observation per line (for CSVs at least).
  return Save(filename, vec, fatal, false, inputSaveType);
}

template<typename eT>
bool Save(const std::string& filename,
          const arma::Row<eT>& rowvec,
          const bool fatal,
          FileType inputSaveType)
{
  return Save(filename, rowvec, fatal, true, inputSaveType);
}

// Save a Sparse Matrix
template<typename eT>
bool Save(const std::string& filename,
          const arma::SpMat<eT>& matrix,
          const bool fatal,
          bool transpose)
{
  MatrixOptions opts;
  opts.Fatal() = fatal;
  opts.NoTranspose() = !transpose;

  return Save(filename, matrix, opts);
}

template<typename eT>
bool Save(const std::string& filename,
          const arma::Mat<eT>& matrix,
          const bool fatal,
          bool transpose,
          FileType inputSaveType)
{
  MatrixOptions opts;
  opts.Fatal() = fatal;
  opts.NoTranspose() = !transpose;
  opts.Format() = inputSaveType;

  return Save(filename, matrix, opts);
}

template<typename MatType, typename DataOptionsType>
bool Save(const std::string& filename,
          const MatType& matrix,
          const DataOptionsBase<DataOptionsType>& opts)
{
  //! just use default copy ctor with = operator and make a copy.
  DataOptionsType copyOpts(opts);
  return Save(filename, matrix, copyOpts);
}

template<typename MatType, typename DataOptionsType>
bool Save(const std::string& filename,
          const MatType& matrix,
          DataOptionsBase<DataOptionsType>& opts)
{
  Timer::Start("saving_data");

  bool success = DetectFileType<MatType>(filename, opts, false);
  if (!success)
  {
    Timer::Stop("saving_data");
    return false;
  }

  std::fstream stream;
  success = OpenFile(filename, opts, false, stream);
  if (!success)
  {
    Timer::Stop("saving_data");
    return false;
  }

  // Try to save the file.
  Log::Info << "Saving " << opts.FileTypeToString() << " to '" << filename
      << "'." << std::endl;
  if constexpr (IsArma<MatType>::value || IsSparseMat<MatType>::value)
  {
    TextOptions txtOpts(std::move(opts));
    if constexpr (IsSparseMat<MatType>::value)
    {
      success = SaveSparse(matrix, txtOpts, filename, stream);
    }
    else if constexpr (IsCol<MatType>::value)
    {
      opts.NoTranspose() = true;
      success = SaveDense(matrix, txtOpts, filename, stream);
    }
    else if constexpr (IsRow<MatType>::value)
    {
      opts.NoTranspose() = false;
      success = SaveDense(matrix, txtOpts, filename, stream);
    }
    else if constexpr (IsDense<MatType>::value)
    {
      success = SaveDense(matrix, txtOpts, filename, stream);
    }
    opts = std::move(txtOpts);
  }
  else if constexpr (!IsArma<MatType>::value && !IsSparseMat<MatType>::value)
  {
    success = SaveModel(matrix, opts, stream);
  }
  else
  {
    if (opts.Fatal())
      Log::Fatal << "DataOptionsType is unknown!  Please use a known type or "
          << "or provide specific overloads." << std::endl;
    else
      Log::Warn << "DataOptionsType is unknown!  Please use a known type or "
          << "or provide specific overloads." << std::endl;

    return false;
  }

  if (!success)
  {
    Timer::Stop("saving_data");
    if (opts.Fatal())
      Log::Fatal << "Save to '" << filename << "' failed." << std::endl;
    else
      Log::Warn << "Save to '" << filename << "' failed." << std::endl;
    return false;
  }

  Timer::Stop("saving_data");

  return success;
}

template<typename eT>
bool SaveDense(const arma::Mat<eT>& matrix,
               TextOptions& opts,
               const std::string& filename,
               std::fstream& stream)
{
  bool success = false;
  arma::Mat<eT> tmp;
  // Transpose the matrix.
  if (!opts.NoTranspose())
  {
    tmp = trans(matrix);
    success = SaveMatrix(tmp, opts, filename, stream);
  }
  else
    success = SaveMatrix(matrix, opts, filename, stream);

  return success;
}

// Save a Sparse Matrix
template<typename eT>
bool SaveSparse(const arma::SpMat<eT>& matrix,
                TextOptions& opts,
                const std::string& filename,
                std::fstream& stream)
{
  bool success = false;
  arma::SpMat<eT> tmp;

  // Transpose the matrix.
  if (!opts.NoTranspose())
  {
    arma::SpMat<eT> tmp = trans(matrix);
    success = SaveMatrix(tmp, opts, filename, stream);
  }
  else
    success = SaveMatrix(matrix, opts, filename, stream);

  return success;
}

template<typename Object>
bool SaveModel(Object& objectToSerialize,
               DataOptionsBase<PlainDataOptions>& opts,
               std::fstream& stream)
{
  try
  {
    if (opts.Format() == FileType::AutoDetect)
    {
      std::stringstream oss;
      oss << "Serialization data format is not specified."
        " Please specify the format to be either BIN, JSON or XML";
      if (opts.Fatal())
        Log::Fatal << oss.str() << std::endl;
      else
        Log::Warn << oss.str() << std::endl;
    }
    else if (opts.Format() == FileType::XML)
    {
      cereal::XMLOutputArchive ar(stream);
      ar(cereal::make_nvp("model", objectToSerialize));
    }
    else if (opts.Format() == FileType::JSON)
    {
      cereal::JSONOutputArchive ar(stream);
      ar(cereal::make_nvp("model", objectToSerialize));
    }
    else if (opts.Format() == FileType::BIN)
    {
      cereal::BinaryOutputArchive ar(stream);
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


// Save a model to file.
// Keep this implementation until mlpack 5. Then we can remove it.
template<typename T>
bool Save(const std::string& filename,
          const std::string& name,
          T& t,
          const bool fatal,
          FileType f,
          std::enable_if_t<HasSerialize<T>::value>*
          )
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
            << " extension? (allowed: xml/bin/json)" << std::endl;
      else
        Log::Warn << "Unable to detect type of '" << filename << "'; save "
            << "failed.  Incorrect extension? (allowed: xml/bin/json)"
            << std::endl;

      return false;
    }
  }

  // Open the file to save to.
  std::ofstream ofs;
#ifdef _WIN32
  if (f == FileType::BIN) // Open non-text types in binary mode on Windows.
    ofs.open(filename, std::ofstream::out | std::ofstream::binary);
  else
    ofs.open(filename, std::ofstream::out);
#else
  ofs.open(filename, std::ofstream::out);
#endif

  if (!ofs.is_open())
  {
    if (fatal)
      Log::Fatal << "Unable to open file '" << filename << "' to save object '"
          << name << "'." << std::endl;
    else
      Log::Warn << "Unable to open file '" << filename << "' to save object '"
          << name << "'." << std::endl;

    return false;
  }

  try
  {
    if (f == FileType::XML)
    {
      cereal::XMLOutputArchive ar(ofs);
      ar(cereal::make_nvp(name.c_str(), t));
    }
    else if (f == FileType::JSON)
    {
      cereal::JSONOutputArchive ar(ofs);
      ar(cereal::make_nvp(name.c_str(), t));
    }
    else if (f == FileType::BIN)
    {
      cereal::BinaryOutputArchive ar(ofs);
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

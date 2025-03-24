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

/*
 * btw, the following two functions are not documented anywhere
 * If they are not exposed to the public API then I can delete them
 * If they are exposed to the public API, then they are deprecated.
 * @rcurtin please comment:
 */
template<typename eT>
[[deprecated("Will be removed in mlpack 5.0.0; use other overloads instead")]]
bool Save(const std::string& filename,
          const arma::Col<eT>& vec,
          const bool fatal,
          FileType inputSaveType)
{
  // Don't transpose: one observation per line (for CSVs at least).
  return Save(filename, vec, fatal, false, inputSaveType);
}

template<typename eT>
[[deprecated("Will be removed in mlpack 5.0.0; use other overloads instead")]]
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
  DataOptions opts;
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
  DataOptions opts;
  opts.Fatal() = fatal;
  opts.NoTranspose() = !transpose;
  opts.FileFormat() = inputSaveType;

  return Save(filename, matrix, opts);
}

//! Save a model to file.
template<typename T>
bool Save(const std::string& filename,
          const std::string& name,
          T& t,
          const bool fatal,
          format f)
{
  ModelOptions opts;
  opts.ObjectName() = name;
  opts.Fatal() = fatal;
  opts.DataFormat() = f;

  return SaveModel(filename, t, opts);
}

template<typename MatType>
bool Save(const std::string& filename,
          const MatType& matrix,
          const DataOptions& opts)
{
  //! just use default copy ctor with = operator and make a copy.
  DataOptions copyOpts = opts;
  return Save(filename, matrix, copyOpts);
}

template<typename MatType, typename DataOptionsType>
bool Save(const std::string& filename,
          const MatType& matrix,
          DataOptions& opts)
{
  Timer::Start("saving_data");

  bool success = DetectFileType(filename, opts, false);
  if (!success)
  {
    Timer::Stop("loading_data");
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
  if (std::is_same_v<DataOptionsType, CSVOptions>)
  {
    if constexpr (IsSparseMat<MatType>::value)
    {
      success = SaveSparse(filename, matrix, opts);
    }
    else if constexpr (IsCol<MatType>::value)
    {
      opts.NoTranspose() = true;
      success = SaveDense(filename, matrix, opts, stream);
    }
    else if constexpr (IsRow<MatType>::value)
    {
      opts.NoTranspose() = false;
      success = SaveDense(filename, matrix, opts, stream);
    }
    else if (IsDense<MatType>::value)
    {
      success = SaveDense(filename, matrix, opts, stream);
    }
  }
  else if (std::is_same_v<DataOptionsType, ImageOptions>)
  {
  }
  else if (std::is_same_v<DataOptionsType, ModelOptions>)
  {
    success = SaveModel(filename, matrix, opts, &stream);
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
bool SaveDense(const std::string& filename,
               const arma::Mat<eT>& matrix,
               DataOptions& opts,
               std::fstream& stream)
{
  bool success = false;
  arma::Mat<eT> tmp;
  // Transpose the matrix.
  if (!opts.NoTranspose())
  {
    tmp = trans(matrix);
    success = SaveMatrix(tmp, opts, stream);
  }
  else
    success = SaveMatrix(matrix, opts, stream);
  
  return success;
}

// Save a Sparse Matrix
template<typename eT>
bool SaveSparse(const std::string& filename,
                const arma::SpMat<eT>& matrix,
                DataOptions& opts,
                std::fstream& stream)
{
  bool success = false;
  arma::SpMat<eT> tmp;
  // Transpose the matrix.
  if (!opts.NoTranspose())
  {
    tmp = trans(matrix);
    success = SaveMatrix(tmp, opts, stream);
  }
  else
    success = SaveMatrix(matrix, opts, stream);

  return success;
}

//! Save a model to file.
template<typename Object>
bool SaveModel(const std::string& filename,
               Object& objectToSerialize,
               ModelOptions& opts,
               std::fstream* stream = nullptr)
{
  try
  {
    if (opts.DataFormat() == format::xml)
    {
      cereal::XMLOutputArchive ar(*stream);
      ar(cereal::make_nvp(opts.ObjectName().c_str(), objectToSerialize));
    }
    else if (opts.DataFormat() == format::json)
    {
      cereal::JSONOutputArchive ar(*stream);
      ar(cereal::make_nvp(opts.ObjectName().c_str(), objectToSerialize));
    }
    else if (opts.DataFormat() == format::binary)
    {
      cereal::BinaryOutputArchive ar(*stream);
      ar(cereal::make_nvp(opts.ObjectName().c_str(), objectToSerialize));
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

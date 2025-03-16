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
  DataOptions opts;
  opts.ObjectName() = name;
  opts.Fatal() = fatal;
  //opts.Model() = t;
  opts.DataFormat() = f;

  return SaveModel(filename, t, opts);
}

template<typename MatType>
bool Save(const std::string& filename,
          const MatType& matrix,
          const DataOptions& opts)
{
  // Copy the options since passing a const into a modifiable function can
  // result in a segmentation fault.
  // We do not copy back to preserve the const property of the function.
  DataOptions copyOpts = opts;
  return Save(filename, matrix, copyOpts);
}

template<typename MatType>
bool Save(const std::string& filename,
          const MatType& matrix,
          DataOptions& opts)
{
  bool success = false;
  using eT = typename MatType::elem_type;
  if constexpr (std::is_same_v<MatType, arma::SpMat<eT>>)
  {
    success = SaveSparse(filename, matrix, opts);
  }
  else if constexpr (std::is_same_v<MatType, arma::Col<eT>>)
  {
    opts.NoTranspose() = true;
    success = SaveDense(filename, matrix, opts);
  }
  else if constexpr (std::is_same_v<MatType, arma::Row<eT>>)
  {
    opts.NoTranspose() = false;
    success = SaveDense(filename, matrix, opts);
  }
  else
  {
    success = SaveDense(filename, matrix, opts);
  }

  return success;
}

template<typename eT>
bool SaveDense(const std::string& filename,
               const arma::Mat<eT>& matrix,
               DataOptions& opts)
{
  Timer::Start("saving_data");
  // Specify that we are Saving.
  opts.Save() = true;
  opts.Load() = false;

  bool success = DetectFileType(filename, opts);
  if (!success)
  {
    Timer::Stop("loading_data");
    return false;
  }
 
  success = OpenFile(filename, opts);
  if (!success)
  {
    Timer::Stop("saving_data");
    return false;
  }

  // Try to save the file.
  Log::Info << "Saving " << opts.FileTypeToString() << " to '" << filename
      << "'." << std::endl;

  arma::Mat<eT> tmp;
  // Transpose the matrix.
  if (!opts.NoTranspose())
  {
    tmp = trans(matrix);
    success = SaveMatrix(tmp, opts);
  }
  else
    success = SaveMatrix(matrix, opts);

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

  // Finally return success.
  return true;
}

// Save a Sparse Matrix
template<typename eT>
bool SaveSparse(const std::string& filename,
                const arma::SpMat<eT>& matrix,
                DataOptions& opts)
{

  Timer::Start("saving_data");
  // Specify that we are Saving.
  opts.Save() = true;
  opts.Load() = false;

  bool success = DetectFileType(filename, opts);
  if (!success)
  {
    Timer::Stop("saving_data");
    return false;
  }

  success = OpenFile(filename, opts);
  if (!success)
  {
    Timer::Stop("saving_data");
    return false;
  }
  
  // Try to save the file.
  Log::Info << "Saving " << opts.FileTypeToString() << " to '" << filename
      << "'." << std::endl;

  arma::SpMat<eT> tmp;
  // Transpose the matrix.
  if (!opts.NoTranspose())
  {
    tmp = trans(matrix);
    success = SaveMatrix(tmp, opts);
  }
  else
    success = SaveMatrix(matrix, opts);

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

  // Finally return success.
  return true;
}

//! Save a model to file.
template<typename Object>
bool SaveModel(const std::string& filename,
               Object& objectToSerialize,
               DataOptions& opts)
{
  bool success = DetectFileType(filename, opts);
  if (!success)
  {
    return false;
  }

  success = OpenFile(filename, opts);
  if (!success)
  {
    return false;
  }

  try
  {
    if (opts.DataFormat() == format::xml)
    {
      cereal::XMLOutputArchive ar(opts.Stream());
      ar(cereal::make_nvp(opts.ObjectName().c_str(), objectToSerialize));
    }
    else if (opts.DataFormat() == format::json)
    {
      cereal::JSONOutputArchive ar(opts.Stream());
      ar(cereal::make_nvp(opts.ObjectName().c_str(), objectToSerialize));
    }
    else if (opts.DataFormat() == format::binary)
    {
      cereal::BinaryOutputArchive ar(opts.Stream());
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

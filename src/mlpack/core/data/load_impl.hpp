/**
 * @file core/data/load_impl.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 * @author Gopi Tatiraju
 *
 * Implementation of templatized load() function defined in load.hpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_IMPL_HPP
#define MLPACK_CORE_DATA_LOAD_IMPL_HPP

// In case it hasn't already been included.
#include "load.hpp"

namespace mlpack {
namespace data {

template<typename MatType, typename DataOptionsType>
bool Load(const std::string& filename,
          MatType& matrix,
          const DataOptionsType& opts,
          const typename std::enable_if_t<
              IsDataOptions<DataOptionsType>::value>*)
{
  DataOptionsType tmpOpts(opts);
  return Load(filename, matrix, tmpOpts);
}

template<typename eT, typename DataOptionsType>
bool Load(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          const DataOptionsType& opts,
          const typename std::enable_if_t<
              IsDataOptions<DataOptionsType>::value>*)
{
  DataOptionsType tmpOpts(opts);
  return Load(files, matrix, tmpOpts);
}

template<typename ObjectType, typename DataOptionsType>
bool Load(const std::string& filename,
          ObjectType& matrix,
          DataOptionsType& opts,
          const typename std::enable_if_t<
              IsDataOptions<DataOptionsType>::value>*)
{
  Timer::Start("loading_data");

  static_assert(!IsArma<ObjectType>::value || !IsSparseMat<ObjectType>::value
      || !HasSerialize<ObjectType>::value, "mlpack can load Armadillo"
      " matrices or serialized mlpack models only; please use a known type.");
  const bool isMatrixType = IsArma<ObjectType>::value ||
      IsSparseMat<ObjectType>::value;
  const bool isSerializable = HasSerialize<ObjectType>::value;
  const bool isSparseMatrixType = IsSparseMat<ObjectType>::value;

  std::fstream stream;
  bool success = OpenFile(filename, opts, true, stream);
  if (!success)
  {
    Timer::Stop("loading_data");
    return false;
  }

  success = DetectFileType<ObjectType>(filename, opts, true, &stream);
  if (!success)
  {
    Timer::Stop("loading_data");
    return false;
  }
  const bool isImageFormat = (opts.Format() == FileType::PNG ||
      opts.Format() == FileType::JPG || opts.Format() == FileType::PNM ||
      opts.Format() == FileType::BMP || opts.Format() == FileType::GIF ||
      opts.Format() == FileType::PSD || opts.Format() == FileType::TGA ||
      opts.Format() == FileType::PIC || opts.Format() == FileType::ImageType);

  if constexpr (isMatrixType)
  {
    if (isImageFormat)
    {
      if constexpr (isSparseMatrixType)
      {
        return HandleError("Cannot load image data into a sparse matrix. "
        "Please use dense matrix instead.", opts);
      }
      else
      {
        ImageOptions imgOpts(std::move(opts));
        std::vector<std::string> files;
        files.push_back(filename);
        success = LoadImage(files, matrix, imgOpts);
        opts = std::move(imgOpts);
      }
    }
    else
    {
      TextOptions txtOpts(std::move(opts));
      success = LoadNumeric(filename, matrix, stream, txtOpts);
      opts = std::move(txtOpts);
    }
  }
  else if constexpr (isSerializable)
  {
    success = LoadModel(matrix, opts, stream);
  }
  else
  {
    return HandleError("DataOptionsType is unknown!  Please use a known type "
        "or provide specific overloads.", opts);
  }

  if (!success)
  {
    Timer::Stop("loading_data");
    std::stringstream oss;
    oss << "Loading from '" << filename << "' failed.";
    return HandleError(oss, opts);
  }
  else
  {
    if constexpr (IsArma<ObjectType>::value)
    {
      Log::Info << "Size is " << matrix.n_rows << " x "
          << matrix.n_cols << ".\n";
    }
  }

  Timer::Stop("loading_data");

  return success;
}

template<typename eT, typename DataOptionsType>
bool Load(const std::vector<std::string>& files,
          arma::Mat<eT>& matrix,
          DataOptionsType& opts,
          const typename std::enable_if_t<
              IsDataOptions<DataOptionsType>::value>*)
{
  bool success = false;
  if (files.empty())
  {
    return HandleError("Load(): given set of filenames is empty;"
        " loading failed.", opts);
  }

  DetectFromExtension<arma::Mat<eT>>(files.back(), opts);
  const bool isImageFormat = (opts.Format() == FileType::PNG ||
      opts.Format() == FileType::JPG || opts.Format() == FileType::PNM ||
      opts.Format() == FileType::BMP || opts.Format() == FileType::GIF ||
      opts.Format() == FileType::PSD || opts.Format() == FileType::TGA ||
      opts.Format() == FileType::PIC || opts.Format() == FileType::ImageType);

  if (isImageFormat)
  {
    ImageOptions imgOpts(std::move(opts));
    success = LoadImage(files, matrix, imgOpts);
    opts = std::move(imgOpts);
  }
  else
  {
    TextOptions txtOpts(std::move(opts));
    success = LoadNumericMultifile(files, matrix, txtOpts);
    opts = std::move(txtOpts);
  }
  return success;
}

template<typename eT>
bool LoadCategorical(const std::string& filename,
                     arma::Mat<eT>& matrix,
                     TextOptions& opts)
{
  // Get the extension and load as necessary.
  Timer::Start("loading_data");

  // Get the extension.
  std::string extension = Extension(filename);
  bool success = false;

  if (extension == "csv" || extension == "tsv" || extension == "txt")
  {
    Log::Info << "Loading '" << filename << "' as CSV dataset.  " << std::flush;
    LoadCSV loader(filename, opts.Fatal());
    success = loader.LoadCategoricalCSV(matrix, opts);
    if (!success)
    {
      Timer::Stop("loading_data");
      return false;
    }
  }
  else if (extension == "arff")
  {
    Log::Info << "Loading '" << filename << "' as ARFF dataset.  "
        << std::flush;
    success = LoadARFF(filename, matrix, opts.DatasetInfo(), opts.Fatal());
    if (!success)
    {
      Timer::Stop("loading_data");
      return false;
    }
    // Retranspose back as we are transposing by default
    if (opts.NoTranspose())
    {
      inplace_trans(matrix);
    }
  }
  else
  {
    // The type is unknown.
    Timer::Stop("loading_data");
    std::stringstream oss;
    oss << "Unable to detect type of '" << filename << "'; "
          << "Incorrect extension?";
    return HandleError(oss, opts);
  }

  Log::Info << "Size is " << matrix.n_rows << " x " << matrix.n_cols << ".\n";

  Timer::Stop("loading_data");

  return true;
}

} // namespace data
} // namespace mlpack

#endif

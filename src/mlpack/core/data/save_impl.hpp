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

namespace mlpack {
namespace data {

template<typename MatType, typename DataOptionsType>
bool Save(const std::string& filename,
          const MatType& matrix,
          const DataOptionsType& opts,
          const typename std::enable_if_t<
              IsDataOptions<DataOptionsType>::value>*)
{
  //! just use default copy ctor with = operator and make a copy.
  DataOptionsType copyOpts(opts);
  return Save(filename, matrix, copyOpts);
}

template<typename ObjectType, typename DataOptionsType>
bool Save(const std::string& filename,
          const ObjectType& matrix,
          DataOptionsType& opts,
          const typename std::enable_if_t<
              IsDataOptions<DataOptionsType>::value>*)
{
  Timer::Start("saving_data");
  static_assert(!IsArma<ObjectType>::value || !IsSparseMat<ObjectType>::value
      || !HasSerialize<ObjectType>::value, "mlpack can save Armadillo"
      " matrices or a serialized mlpack model only; please use a known type.");
  const bool isMatrixType = IsArma<ObjectType>::value ||
      IsSparseMat<ObjectType>::value;
  const bool isSerializable = HasSerialize<ObjectType>::value;
  const bool isSparseMatrixType = IsSparseMat<ObjectType>::value;

  bool success = DetectFileType<ObjectType>(filename, opts, false);
  if (!success)
  {
    Timer::Stop("saving_data");
    return false;
  }

  const bool isImageFormat = (opts.Format() == FileType::PNG ||
      opts.Format() == FileType::JPG || opts.Format() == FileType::PNM ||
      opts.Format() == FileType::BMP || opts.Format() == FileType::GIF ||
      opts.Format() == FileType::PSD || opts.Format() == FileType::TGA ||
      opts.Format() == FileType::PIC || opts.Format() == FileType::ImageType);

  std::fstream stream;
  if (!isImageFormat)
  {
    success = OpenFile(filename, opts, false, stream);
    if (!success)
    {
      Timer::Stop("saving_data");
      return false;
    }
  }

  // Try to save the file.
  Log::Info << "Saving " << opts.FileTypeToString() << " to '" << filename
      << "'." << std::endl;
  if constexpr (isMatrixType)
  {
    if (isImageFormat)
    {
      if constexpr (isSparseMatrixType)
      {
        arma::Mat<typename ObjectType::elem_type> tmp =
            arma::conv_to<arma::Mat<
            typename ObjectType::elem_type>>::from(matrix);
        ImageOptions imgOpts(std::move(opts));
        std::vector<std::string> files;
        files.push_back(filename);
        success = SaveImage(files, tmp, imgOpts);
        opts = std::move(imgOpts);
      }
      else
      {
        ImageOptions imgOpts(std::move(opts));
        std::vector<std::string> files;
        files.push_back(filename);
        success = SaveImage(files, matrix, imgOpts);
        opts = std::move(imgOpts);
      }
    }
    else
    {
      success = SaveNumeric(filename, matrix, stream, opts);
    }
  }
  else if constexpr (isSerializable)
  {
    success = SaveModel(matrix, opts, stream);
  }
  else
  {
    return HandleError("DataOptionsType is unknown!  Please use a known type "
        "or provide specific overloads.", opts);
  }

  if (!success)
  {
    Timer::Stop("saving_data");
    std::stringstream oss;
    oss << "Save to '" << filename << "' failed.";
    return HandleError(oss, opts);
  }

  Timer::Stop("saving_data");

  return success;
}

} // namespace data
} // namespace mlpack

#endif

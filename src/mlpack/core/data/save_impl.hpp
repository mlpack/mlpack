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
bool SaveNumeric(const std::string& filename,
                 const ObjectType& matrix,
                 std::fstream& stream,
                 DataOptionsBase<DataOptionsType>& opts)
{
  bool success = false;

  TextOptions txtOpts(std::move(opts));
  if constexpr (IsSparseMat<ObjectType>::value)
  {
    success = SaveSparse(matrix, txtOpts, filename, stream);
  }
  else if constexpr (IsCol<ObjectType>::value)
  {
    opts.NoTranspose() = true;
    success = SaveDense(matrix, txtOpts, filename, stream);
  }
  else if constexpr (IsRow<ObjectType>::value)
  {
    opts.NoTranspose() = false;
    success = SaveDense(matrix, txtOpts, filename, stream);
  }
  else if constexpr (IsDense<ObjectType>::value)
  {
    success = SaveDense(matrix, txtOpts, filename, stream);
  }
  opts = std::move(txtOpts);

  return success;
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
               const DataOptionsBase<PlainDataOptions>& opts,
               std::fstream& stream)
{
  try
  {
    if (opts.Format() == FileType::XML)
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

template<typename MatType, typename DataOptionsType>
bool SaveMatrix(const MatType& matrix,
                const DataOptionsType& opts,
#ifdef ARMA_USE_HDF5
                const std::string& filename,
#else
                const std::string& /* filename */,
#endif
                std::fstream& stream)
{
  bool success = false;
  if (opts.Format() == FileType::HDF5Binary)
  {
#ifdef ARMA_USE_HDF5
    // We can't save with streams for HDF5.
    success = matrix.save(filename, opts.ArmaFormat());
#endif
  }
  else
  {
    success = matrix.save(stream, opts.ArmaFormat());
  }
  return success;
}

} // namespace data
} // namespace mlpack

#endif

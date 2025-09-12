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

  std::fstream stream;
  success = OpenFile(filename, opts, false, stream);
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
        SaveImage(files, tmp, imgOpts);
        opts = std::move(imgOpts);
      }
      else
      {
        ImageOptions imgOpts(std::move(opts));
        std::vector<std::string> files;
        files.push_back(filename);
        SaveImage(files, matrix, imgOpts);
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

// Save a model to file.
// Keep this implementation until mlpack 5. Then we can remove it.
template<typename T>
bool Save(const std::string& filename,
          const std::string& name,
          T& t,
          const bool fatal,
          format f,
          std::enable_if_t<HasSerialize<T>::value>*)
{
  if (f == format::autodetect)
  {
    std::string extension = Extension(filename);

    if (extension == "xml")
      f = format::xml;
    else if (extension == "bin")
      f = format::binary;
    else if (extension == "json")
      f = format::json;
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
  if (f == format::binary) // Open non-text types in binary mode on Windows.
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
    if (f == format::xml)
    {
      cereal::XMLOutputArchive ar(ofs);
      ar(cereal::make_nvp(name.c_str(), t));
    }
    else if (f == format::json)
    {
      cereal::JSONOutputArchive ar(ofs);
      ar(cereal::make_nvp(name.c_str(), t));
    }
    else if (f == format::binary)
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

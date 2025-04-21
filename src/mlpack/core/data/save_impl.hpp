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
          const DataOptionsType& opts)
{
  //! just use default copy ctor with = operator and make a copy.
  DataOptionsType copyOpts(opts);
  return Save(filename, matrix, copyOpts);
}

template<typename MatType, typename DataOptionsType>
bool Save(const std::string& filename,
          const MatType& matrix,
          DataOptionsType& opts)
{
  Timer::Start("saving_data");

  bool success = DetectFileType(filename, opts, false);
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
  if constexpr (std::is_same_v<DataOptionsType, CSVOptions>)
  {
    CSVOptions csvOpts(opts);
    if constexpr (IsSparseMat<MatType>::value)
    {
      success = SaveSparse(matrix, csvOpts, stream);
    }
    else if constexpr (IsCol<MatType>::value)
    {
      opts.NoTranspose() = true;
      success = SaveDense(matrix, csvOpts, stream);
    }
    else if constexpr (IsRow<MatType>::value)
    {
      opts.NoTranspose() = false;
      success = SaveDense(matrix, csvOpts, stream);
    }
    else if constexpr (IsDense<MatType>::value)
    {
      success = SaveDense(matrix, csvOpts, stream);
    }
  }
  else if constexpr (std::is_same_v<DataOptionsType, ModelOptions>)
  {
    ModelOptions modOpts(opts);
    success = SaveModel(matrix, modOpts, stream);
  }
  else
  {
    throw std::runtime_error("DataOptionType is unknown!."
        "please use a known type or provide specific overloads");
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
               CSVOptions& opts,
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
bool SaveSparse(const arma::SpMat<eT>& matrix,
                CSVOptions& opts,
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
bool SaveModel(Object& objectToSerialize,
               ModelOptions& opts,
               std::fstream& stream)
{
  try
  {
    if (opts.DataFormat() == format::xml)
    {
      cereal::XMLOutputArchive ar(stream);
      ar(cereal::make_nvp(opts.ObjectName().c_str(), objectToSerialize));
    }
    else if (opts.DataFormat() == format::json)
    {
      cereal::JSONOutputArchive ar(stream);
      ar(cereal::make_nvp(opts.ObjectName().c_str(), objectToSerialize));
    }
    else if (opts.DataFormat() == format::binary)
    {
      cereal::BinaryOutputArchive ar(stream);
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

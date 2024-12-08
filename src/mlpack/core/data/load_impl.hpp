/**
 * @file core/data/load_impl.hpp
 * @author Ryan Curtin
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

#include <algorithm>
#include <exception>

#include "extension.hpp"
#include "detect_file_type.hpp"
#include "string_algorithms.hpp"

namespace mlpack {
namespace data {

namespace details{

template<typename Tokenizer>
std::vector<std::string> ToTokens(Tokenizer& lineTok)
{
  std::vector<std::string> tokens;
  std::transform(std::begin(lineTok), std::end(lineTok),
                 std::back_inserter(tokens),
                 [&tokens](std::string const &str)
  {
    std::string trimmedToken(str);
    Trim(trimmedToken);
    return std::move(trimmedToken);
  });

  return tokens;
}

inline
void TransposeTokens(std::vector<std::vector<std::string>> const &input,
                     std::vector<std::string>& output,
                     size_t index)
{
  output.clear();
  for (size_t i = 0; i != input.size(); ++i)
  {
    output.emplace_back(input[i][index]);
  }
}

} // namespace details

inline
bool FileExist(std::fstream& stream)
{
#ifdef  _WIN32 // Always open in binary mode on Windows.
  stream.open(filename.c_str(), std::fstream::in | std::fstream::binary);
#else
  stream.open(filename.c_str(), std::fstream::in);
#endif
  if (!stream.is_open())
  {
    Timer::Stop("loading_data");
    if (fatal)
      Log::Fatal << "Cannot open file '" << filename << "'. " << std::endl;
    else
      Log::Warn << "Cannot open file '" << filename << "'; load failed."
          << std::endl;

    return false;
  }
  return true;
}

inline
bool DetectFile(const std::fstream& stream,
                FileType& loadType
                const LoadOptions& opts);
{
  if (opts.FileType() == FileType::AutoDetect)
  {
    // Attempt to auto-detect the type from the given file.
    loadType = AutoDetect(stream, filename);
    // Provide error if we don't know the type.
    if (loadType == FileType::FileTypeUnknown)
    {
      Timer::Stop("loading_data");
      if (opts.Fatal())
        Log::Fatal << "Unable to detect type of '" << filename << "'; "
            << "incorrect extension?" << std::endl;
      else
        Log::Warn << "Unable to detect type of '" << filename << "'; load "
            << "failed. Incorrect extension?" << std::endl;

      return false;
    }
  }
  return true;
}

/**
 * Similar functionality that exists in Pandas to handle NaN values.
 * The following functions define a backward fill policy to take the last value
 * in a column before a NaN, and replace all the following NaN with the last
 * value in Backward manner
 *
 * @param matrix The dataset provided as armadillo matrix
 */
template<typename eT>
inline
void FillBackward(arma::Mat<eT>& matrix)
{
  eT lastValue = std::numeric_limits<eT>::signaling_NaN();
  for (size_t j = 0; j < matrix.n_cols; ++j)
  {
    size_t i = (matrix.n_rows - 1);
    for (i > -1; --i;)
    {
      if (std::isnan(matrix.at(i, j)))
      {
        matrix.at(i, j) = lastValue;
      }
      else
      {
        lastValue = matrix.at(i, j);
      }
    }
  }
}

/**
 * Similar functionality that exists in Pandas to handle NaN values.
 * The following functions define a forward fill policy to take the last value
 * in a column before a NaN, and replace all the following NaN with the last
 * value in Forward manner
 *
 * @param matrix The dataset provided as armadillo matrix
 */
template<typename eT>
inline
void FillForward(arma::Mat<eT>& matrix)
{
  double lastValue = std::numeric_limits<double>::signaling_NaN();
  for (size_t j = 0; j < matrix.n_cols; ++j)
  {
    for (size_t i = 0; i < matrix.n_rows; ++i)
    {
      if (std::isnan(matrix.at(i, j)))
      {
        matrix.at(i, j) = lastValue;
      }
      else
      {
        lastValue = matrix.at(i, j);
      }
    }
  }
}

inline
bool LoadCSVASCII(const std::string& filename,
                  arma::Mat<eT>& matrix,
                  const LoadOptions& opts)
{
  if (opts.Transpose() && opts.HasHeaders() &&
      !opts.SemiColon() && !opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, opts.Headers(),
          arma::csv_opts::trans), arma::csv_ascii);
  
  else if (opts.Transpose() && !opts.HasHeaders() &&
           !opts.SemiColon() && !opts.MissingToNan())
 
    success = matrix.load(arma::csv_name(filename, arma::csv_opts::trans),
        arma::csv_ascii);
  
  else if (!opts.Transpose() && opts.HasHeaders() &&
           !opts.SemiColon() && !opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, opts.Headers()),
        arma::csv_ascii);
  
  else if (!opts.Transpose() && !opts.HasHeaders() &&
           opts.SemiColon() && !opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, arma::csv_opts::semicolon),
        arma::csv_ascii);

  else if (!opts.Transpose() && !opts.HasHeaders() &&
           !opts.SemiColon() && opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, arma::csv_opts::strict),
        arma::csv_ascii);

  else if (opts.Transpose() && opts.HasHeaders() &&
           opts.SemiColon() && !opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, opts.Headers(),
          arma::csv_opts::semicolon + arma::csv_opts::trans), arma::csv_ascii);

  else if (opts.Transpose() && opts.HasHeaders() &&
           opts.SemiColon() && opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, opts.Headers(),
          arma::csv_opts::semicolon + arma::csv_opts::trans +
          arma::csv_opts::strict), arma::csv_ascii);

  else if (opts.Transpose() && !opts.HasHeaders() &&
           !opts.SemiColon() && opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename,
          arma::csv_opts::trans + arma::csv_opts::strict), arma::csv_ascii);

  else if (opts.Transpose() && !opts.HasHeaders() &&
           opts.SemiColon() && !opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename,
          arma::csv_opts::trans + arma::csv_opts::semicolon), arma::csv_ascii);

  else if (!opts.Transpose() && opts.HasHeaders() &&
            opts.SemiColon() && !opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, opts.Headers(),
          arma::csv_opts::semicolon), arma::csv_ascii);

  else if (!opts.Transpose() && opts.HasHeaders() &&
           !opts.SemiColon() && opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, opts.Headers(),
          arma::csv_opts::strict), arma::csv_ascii);

  else if (!opts.Transpose() && opts.HasHeaders() &&
            opts.SemiColon() && opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, opts.Headers(),
        arma::csv_opts::strict + arma::csv_opts::semicolon), arma::csv_ascii);

  else if (opts.Transpose() && !opts.HasHeaders() &&
           opts.SemiColon() && opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, arma::csv_opts::strict +
          arma::csv_opts::trans + arma::csv_opts::semicolon), arma::csv_ascii);

  else if (!opts.Transpose() && !opts.HasHeaders() &&
           opts.SemiColon() && opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, arma::csv_opts::strict +
          arma::csv_opts::semicolon), arma::csv_ascii);

  // Detect if NANs exist and replace according to the policy.
  // For now this function require transpose. As we can not be sure,
  // if the user always asked for tranpose. While I think that we need to have
  // the function on a transposed situation and then change it back
  if (opts.FillForward())

    FillForward();
  else if (opts.FillBackward())
    FillBackward();

  return success;
}

inline
bool LoadHDF5(const std::string& filename,
              arma::Mat<eT>& matrix,
              LoadOptions& opts)
{
#ifndef ARMA_USE_HDF5
    // Ensure that HDF5 is supported.
    Timer::Stop("loading_data");
    if (opts.Fatal())
      Log::Fatal << "Attempted to load '" << filename << "' as HDF5 data, but "
          << "Armadillo was compiled without HDF5 support.  Load failed."
          << std::endl;
    else
      Log::Warn << "Attempted to load '" << filename << "' as HDF5 data, but "
          << "Armadillo was compiled without HDF5 support.  Load failed."
          << std::endl;

    return false;
#endif
    
    //success = matrix.load(filename, ToArmaFileType(loadType));
}

template<typename eT>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          const bool fatal,
          const bool transpose,
          const FileType inputLoadType)
{
  LoadOptions& opts;
  opts.Fatal() = fatal;
  opts.Transpose() = transpose;
  opts.FileType() = inputLoadType;

  Load(filename, matrix, opts);
}

template<typename eT>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          LoadOptions& opts)
 {
  Timer::Start("loading_data");
  bool success;
  std::fstream stream;
  std::string stringType;
 
  success = FileExist(stream);
  if (!success)
    return false;

  FileType loadType = opts.FileType();

  success = DetectFileType(stream, loadType, opts);
  if (!success)
    return false;

  stringType = GetStringType(loadType);

  if (loadType != FileType::RawBinary)
    Log::Info << "Loading '" << filename << "' as " << stringType << ".  "
        << std::flush;

  // We can't use the stream if the type is HDF5.
  if (loadType == FileType::HDF5Binary)
  {
    success = LoadHDSF5(filename, matrix, opts);
  }
  else if (loadType == FileType::CSVASCII)
  {
    success = LoadCSVASCII(filename, matrix, opts);
  }
  else
  {
    if (loadType == FileType::RawBinary)
      Log::Warn << "Loading '" << filename << "' as " << stringType << "; " 
        << "but this may not be the actual filetype!" << std::endl;

    success = matrix.load(stream, ToArmaFileType(loadType));
    if (opts.Transpose())
      inplace_trans(matrix);
  }

  if (!success)
  {
    Log::Info << std::endl;
    Timer::Stop("loading_data");
    if (opts.Fatal())
      Log::Fatal << "Loading from '" << filename << "' failed." << std::endl;
    else
      Log::Warn << "Loading from '" << filename << "' failed." << std::endl;

    return false;
  }
  else
    Log::Info << "Size is " << matrix.n_rows << " x " << matrix.n_cols << ".\n";

  Timer::Stop("loading_data");

  // Finally, return the success indicator.
  return success;
}

// Load with mappings.  Unfortunately we have to implement this ourselves.
template<typename eT, typename PolicyType>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          DatasetMapper<PolicyType>& info,
          const bool fatal,
          const bool transpose)
{
  // Get the extension and load as necessary.
  Timer::Start("loading_data");

  bool sucess;
  std::fstream stream;
  success = FileExist(stream);
  if (!success)
    return false;

  // Get the extension.
  std::string extension = Extension(filename);

  if (extension == "csv" || extension == "tsv" || extension == "txt")
  {
    Log::Info << "Loading '" << filename << "' as CSV dataset.  " << std::flush;
    try
    {
      LoadCSV loader(filename);
      loader.LoadCategoricalCSV(matrix, info, transpose);
    }
    catch (std::exception& e)
    {
      Timer::Stop("loading_data");
      if (fatal)
        Log::Fatal << e.what() << std::endl;
      else
        Log::Warn << e.what() << std::endl;

      return false;
    }
  }
  else if (extension == "arff")
  {
    Log::Info << "Loading '" << filename << "' as ARFF dataset.  "
        << std::flush;
    try
    {
      LoadARFF(filename, matrix, info);

      // We transpose by default.  So, un-transpose if necessary...
      if (!transpose)
      {
        inplace_trans(matrix);
      }
    }
    catch (std::exception& e)
    {
      Timer::Stop("loading_data");
      if (fatal)
        Log::Fatal << e.what() << std::endl;
      else
        Log::Warn << e.what() << std::endl;

      return false;
    }
  }
  else
  {
    // The type is unknown.
    Timer::Stop("loading_data");
    if (fatal)
      Log::Fatal << "Unable to detect type of '" << filename << "'; "
          << "incorrect extension?" << std::endl;
    else
      Log::Warn << "Unable to detect type of '" << filename << "'; load failed."
          << " Incorrect extension?" << std::endl;

    return false;
  }

  Log::Info << "Size is " << matrix.n_rows << " x " << matrix.n_cols << ".\n";

  Timer::Stop("loading_data");

  return true;
}

// For loading data into sparse matrix
template <typename eT>
bool Load(const std::string& filename,
          arma::SpMat<eT>& matrix,
          const bool fatal,
          const bool transpose,
          const FileType inputLoadType)
{
  LoadOptions opts;
  opts.Fatal() = fatal;
  opts.Transpose() = transpose;
  opts.FileType() = inputLoadType;

  Load(filename, matrix, opts);
}

template <typename eT>
bool Load(const std::string& filename,
          arma::SpMat<eT>& matrix,
          LoadOptions& opts)
{

  Timer::Start("loading_data");
  bool success;
  std::fstream stream;
  FileType loadType = opts.FileType();
  
  success = FileExist(stream);
  if (!success)
    return false;
  
  success = DetectFileType(stream, loadType, opts);
  if (!success)
    return false;

  // There is still a small amount of differentiation that needs to be done:
  // if we got a text type, it could be a coordinate list.  We will make an
  // educated guess based on the shape of the input.
  if (loadType == FileType::RawASCII)
  {
    // Get the number of columns in the file.  If it is the right shape, we
    // will assume it is sparse.
    const size_t cols = CountCols(stream);
    if (cols == 3)
    {
      // We have the right number of columns, so assume the type is a
      // coordinate list.
      loadType = FileType::CoordASCII;
    }
  }

  // Filter out invalid types.
  if ((loadType == FileType::PGMBinary) ||
      (loadType == FileType::PPMBinary) ||
      (loadType == FileType::ArmaASCII) ||
      (loadType == FileType::RawBinary))
  {
    if (opts.Fatal())
      Log::Fatal << "Cannot load '" << filename << "' with type "
          << GetStringType(loadType) << " into a sparse matrix; format is "
          << "only supported for dense matrices." << std::endl;
    else
      Log::Warn << "Cannot load '" << filename << "' with type "
          << GetStringType(loadType) << " into a sparse matrix; format is "
          << "only supported for dense matrices; load failed." << std::endl;

    return false;
  }
  else if (loadType == FileType::CSVASCII)
  {
    // Armadillo sparse matrices can't load CSVs, so we have to load a separate
    // matrix to do that.  If the CSV has three columns, we assume it's a
    // coordinate list.
    arma::Mat<eT> dense;
    success = dense.load(stream, ToArmaFileType(loadType));
    if (dense.n_cols == 3)
    {
      arma::umat locations = arma::conv_to<arma::umat>::from(
          dense.cols(0, 1).t());
      matrix = arma::SpMat<eT>(locations, dense.col(2));
    }
    else
    {
      matrix = arma::conv_to<arma::SpMat<eT>>::from(dense);
    }
  }
  else
  {
    success = matrix.load(stream, ToArmaFileType(loadType));
  }
  
  if (opts.transpose())
  {
    inplace_trans(matrix);
  }

  if (!success)
  {
    Log::Info << std::endl;
    Timer::Stop("loading_data");
    if (fatal)
      Log::Fatal << "Loading from '" << filename << "' failed." << std::endl;
    else
      Log::Warn << "Loading from '" << filename << "' failed." << std::endl;

    return false;
  }
  else
  {
    Log::Info << "Size is " << matrix.n_rows << " x " << matrix.n_cols << ".\n";
  }

  Timer::Stop("loading_data");

  // Finally, return the success indicator.
  return success;
}

} // namespace data
} // namespace mlpack

#endif

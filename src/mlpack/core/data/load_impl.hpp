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

#include <algorithm>
#include <exception>

#include "extension.hpp"
#include "load_utilities.hpp"
#include "string_algorithms.hpp"


namespace mlpack {
namespace data {

template<typename eT>
bool LoadCSVASCII(const std::string& filename,
                  arma::Mat<eT>& matrix,
                  const DataOptions& opts)
{
  bool success = false;
  if (!opts.NoTranspose() && opts.HasHeaders() &&
      !opts.SemiColon() && !opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, opts.Headers(),
          arma::csv_opts::trans), arma::csv_ascii);
  
  else if (!opts.NoTranspose() && !opts.HasHeaders() &&
           !opts.SemiColon() && !opts.MissingToNan())
 
    success = matrix.load(arma::csv_name(filename, arma::csv_opts::trans),
        arma::csv_ascii);
  
  else if (opts.NoTranspose() && !opts.HasHeaders() &&
           !opts.SemiColon() && !opts.MissingToNan())

    success = matrix.load(filename, arma::csv_ascii);

  else if (opts.NoTranspose() && opts.HasHeaders() &&
           !opts.SemiColon() && !opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, opts.Headers()),
        arma::csv_ascii);
  
  else if (opts.NoTranspose() && !opts.HasHeaders() &&
           opts.SemiColon() && !opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, arma::csv_opts::semicolon),
        arma::csv_ascii);

  else if (opts.NoTranspose() && !opts.HasHeaders() &&
           !opts.SemiColon() && opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, arma::csv_opts::strict),
        arma::csv_ascii);

  else if (!opts.NoTranspose() && opts.HasHeaders() &&
           opts.SemiColon() && !opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, opts.Headers(),
          arma::csv_opts::semicolon + arma::csv_opts::trans), arma::csv_ascii);

  else if (!opts.NoTranspose() && opts.HasHeaders() &&
           opts.SemiColon() && opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, opts.Headers(),
          arma::csv_opts::semicolon + arma::csv_opts::trans +
          arma::csv_opts::strict), arma::csv_ascii);

  else if (!opts.NoTranspose() && !opts.HasHeaders() &&
           !opts.SemiColon() && opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename,
          arma::csv_opts::trans + arma::csv_opts::strict), arma::csv_ascii);

  else if (!opts.NoTranspose() && !opts.HasHeaders() &&
           opts.SemiColon() && !opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename,
          arma::csv_opts::trans + arma::csv_opts::semicolon), arma::csv_ascii);

  else if (opts.NoTranspose() && opts.HasHeaders() &&
           opts.SemiColon() && !opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, opts.Headers(),
          arma::csv_opts::semicolon), arma::csv_ascii);

  else if (opts.NoTranspose() && opts.HasHeaders() &&
           !opts.SemiColon() && opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, opts.Headers(),
          arma::csv_opts::strict), arma::csv_ascii);

  else if (opts.NoTranspose() && opts.HasHeaders() &&
            opts.SemiColon() && opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, opts.Headers(),
        arma::csv_opts::strict + arma::csv_opts::semicolon), arma::csv_ascii);

  else if (!opts.NoTranspose() && !opts.HasHeaders() &&
           opts.SemiColon() && opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, arma::csv_opts::strict +
          arma::csv_opts::trans + arma::csv_opts::semicolon), arma::csv_ascii);

  else if (opts.NoTranspose() && !opts.HasHeaders() &&
           opts.SemiColon() && opts.MissingToNan())

    success = matrix.load(arma::csv_name(filename, arma::csv_opts::strict +
          arma::csv_opts::semicolon), arma::csv_ascii);

  return success;
}

template<typename eT>
bool LoadHDF5(const std::string& filename,
              arma::Mat<eT>& matrix,
              const DataOptions& opts)
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
  
  return matrix.load(filename, ToArmaFileType(opts.FileFormat()));
}

// The following functions are kept for backward compatibility,
// Please remove them when we release mlpack 5.
template<typename eT>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          const bool fatal,
          const bool transpose,
          const FileType inputLoadType)
{
  DataOptions opts;
  opts.Fatal() = fatal;
  opts.NoTranspose() = !transpose;
  opts.FileFormat() = inputLoadType;

  return Load(filename, matrix, opts);
}

// For loading data into sparse matrix
template <typename eT>
bool Load(const std::string& filename,
          arma::SpMat<eT>& matrix,
          const bool fatal,
          const bool transpose,
          const FileType inputLoadType)
{
  DataOptions opts;
  opts.Fatal() = fatal;
  opts.NoTranspose() = !transpose;
  opts.FileFormat() = inputLoadType;

  return Load(filename, matrix, opts);
}

// For loading data into a column vector
template <typename eT>
bool Load(const std::string& filename,
          arma::Col<eT>& vec,
          const bool fatal)
{
  DataOptions opts;
  opts.Fatal() = fatal;
  return Load(filename, vec, opts);
}

// For loading data into a raw vector
template <typename eT>
bool Load(const std::string& filename,
          arma::Row<eT>& rowvec,
          const bool fatal)
{
  DataOptions opts;
  opts.Fatal() = fatal;
  return Load(filename, rowvec, opts);
}

// Load with mappings.  Unfortunately we have to implement this ourselves.
template<typename eT, typename PolicyType>
bool Load(const std::string& filename,
          arma::Mat<eT>& matrix,
          DatasetMapper<PolicyType>& info,
          const bool fatal,
          const bool transpose)
{
  DataOptions opts;
  opts.Fatal() = fatal;
  opts.NoTranspose() = !transpose;
  // @rcurtin just commenting this one as we need to find a solution for the
  // template parameter of the DatasetMapper. Assigning the following variable
  // will simply fails as `info` is different type from opts.Mapper()
  //opts.Mapper() = info;

  return Load(filename, matrix, opts);
}

template<typename MatType>
bool Load(const std::string& filename,
          MatType& matrix,
          const DataOptions& opts)
{
  // Copy the options since passing a const into a modifiable function can
  // result in a segmentation fault.
  // We do not copy back to preserve the const property of the function.
  DataOptions copyOpts = opts;
  return Load(filename, matrix, copyOpts);
}
// This is the function that the user is supposed to call.
// Other functions of this class should be labelled as private.
template<typename MatType>
bool Load(const std::string& filename,
          MatType& matrix,
          DataOptions& opts)
{
  using eT = typename MatType::elem_type;
  Timer::Start("loading_data");
  bool success;
  std::fstream stream;
  std::string stringType;
 
  success = FileExist(filename, stream, opts);
  if (!success)
    return false;

  FileType loadType = opts.FileFormat();

  success = DetectFileType(filename, stream, loadType, opts);
  if (!success)
    return false;

  // Update the FileFormat after detecting it.
  // Probably better to merge the entire operation
  // inside the DetectFiltType @rcurtin will have better idea than me
  // regarding this.
  opts.FileFormat() = loadType;

  if constexpr (std::is_same_v<MatType, arma::SpMat<eT>>)
  {
    success = LoadSparse(filename, stream, matrix, opts);
  }
  else if (opts.Categorical())
  {
    success = LoadCategorical(filename, matrix, opts);
  }
  else if constexpr (MatType::is_col)
  {
    std::cout << MatType::is_col << std::endl;
    success = LoadCol(filename, matrix, opts);
  }
  else if constexpr (MatType::is_row)
  {
    success = LoadRow(filename, matrix, opts);
  }
  else if constexpr (std::is_same_v<MatType, arma::Mat<eT>>) 
  {
    success = LoadDense(filename, stream, matrix, opts);
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

template<typename eT>
bool LoadDense(const std::string& filename,
               std::fstream& stream,
               arma::Mat<eT>& matrix,
               DataOptions& opts)
{
  bool success;
  if (opts.FileFormat() != FileType::RawBinary)
    Log::Info << "Loading '" << filename << "' as " 
        << opts.FileTypeToString() << ".  " << std::flush;

  // We can't use the stream if the type is HDF5.
  if (opts.FileFormat() == FileType::HDF5Binary)
  {
    success = LoadHDF5(filename, matrix, opts);
  }
  else if (opts.FileFormat() == FileType::CSVASCII)
  {
    success = LoadCSVASCII(filename, matrix, opts);
  }
  else
  {
    if (opts.FileFormat() == FileType::RawBinary)
      Log::Warn << "Loading '" << filename << "' as " 
        << opts.FileTypeToString() << "; " 
        << "but this may not be the actual filetype!" << std::endl;

    success = matrix.load(stream, ToArmaFileType(opts.FileFormat()));
    if (!opts.NoTranspose())
      inplace_trans(matrix);
  }
 return success;
}

template <typename eT>
bool LoadSparse(const std::string& filename,
                std::fstream& stream,
                arma::SpMat<eT>& matrix,
                DataOptions& opts)
{
  bool success;
  // There is still a small amount of differentiation that needs to be done:
  // if we got a text type, it could be a coordinate list.  We will make an
  // educated guess based on the shape of the input.
  if (opts.FileFormat() == FileType::RawASCII)
  {
    // Get the number of columns in the file.  If it is the right shape, we
    // will assume it is sparse.
    const size_t cols = CountCols(stream);
    if (cols == 3)
    {
      // We have the right number of columns, so assume the type is a
      // coordinate list.
      opts.FileFormat() = FileType::CoordASCII;
    }
  }

  // Filter out invalid types.
  if ((opts.FileFormat() == FileType::PGMBinary) ||
      (opts.FileFormat() == FileType::PPMBinary) ||
      (opts.FileFormat() == FileType::ArmaASCII) ||
      (opts.FileFormat() == FileType::RawBinary))
  {
    if (opts.Fatal())
      Log::Fatal << "Cannot load '" << filename << "' with type "
          << opts.FileTypeToString() << " into a sparse matrix; format is "
          << "only supported for dense matrices." << std::endl;
    else
      Log::Warn << "Cannot load '" << filename << "' with type "
          << opts.FileTypeToString() << " into a sparse matrix; format is "
          << "only supported for dense matrices; load failed." << std::endl;

    return false;
  }
  else if (opts.FileFormat() == FileType::CSVASCII)
  {
    // Armadillo sparse matrices can't load CSVs, so we have to load a separate
    // matrix to do that.  If the CSV has three columns, we assume it's a
    // coordinate list.
    arma::Mat<eT> dense;
    success = dense.load(stream, ToArmaFileType(opts.FileFormat()));
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
    success = matrix.load(stream, ToArmaFileType(opts.FileFormat()));
  }
  
  if (!opts.NoTranspose())
  {
    // It seems that there is no direct way to use inplace_trans() on 
    // sparse matrices. Therefore, we need the temporary SpMat
    arma::SpMat<eT> tmp = matrix.t();
    matrix = tmp;
  }

  return success;
}

template<typename eT>
bool LoadCategorical(const std::string& filename,
                     arma::Mat<eT>& matrix,
                     DataOptions& opts)
{
  // Get the extension and load as necessary.
  Timer::Start("loading_data");

  // Get the extension.
  std::string extension = Extension(filename);
  bool success = false;

  if (extension == "csv" || extension == "tsv" || extension == "txt")
  {
    Log::Info << "Loading '" << filename << "' as CSV dataset.  " << std::flush;
    LoadCSV loader(filename);
    // would be smarter to pass DataOptions directly in here,
    // @rcurtin what do you think ? should I refactor?
    success = loader.LoadCategoricalCSV(matrix, opts.Mapper(), !opts.NoTranspose());
    if (!success)
    {
      return false;
      Timer::Stop("loading_data");
    }
  }
  else if (extension == "arff")
  {
    Log::Info << "Loading '" << filename << "' as ARFF dataset.  "
        << std::flush;
    success = LoadARFF(filename, matrix, opts.Mapper());
    if (!success)
    {
      return false;
      Timer::Stop("loading_data");
    }
    if (!opts.NoTranspose())
    {
      inplace_trans(matrix);
    }
  }
  else
  {
    // The type is unknown.
    Timer::Stop("loading_data");
    if (opts.Fatal())
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

} // namespace data
} // namespace mlpack

#endif

/**
 * @file bindings/cli/print_type_doc_impl.hpp
 * @author Ryan Curtin
 *
 * Print documentation for a given type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_CLI_PRINT_TYPE_DOC_IMPL_HPP
#define MLPACK_BINDINGS_CLI_PRINT_TYPE_DOC_IMPL_HPP

#include "print_type_doc.hpp"

namespace mlpack {
namespace bindings {
namespace cli {

/**
 * Return a string representing the command-line type of an option.
 */
template<typename T>
std::string PrintTypeDoc(
    util::ParamData& data,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type*,
    const typename std::enable_if<!util::IsStdVector<T>::value>::type*,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type*,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  // A flag type.
  if (std::is_same<T, bool>::value)
  {
    return "A boolean flag option.  If not specified, it is false; if "
        "specified, it is true.";
  }
  // An integer.
  else if (std::is_same<T, int>::value)
  {
    return "An integer (i.e., \"1\").";
  }
  // A floating point value.
  else if (std::is_same<T, double>::value)
  {
    return "A floating-point number (i.e., \"0.5\").";
  }
  // A string.
  else if (std::is_same<T, std::string>::value)
  {
    return "A character string (i.e., \"hello\").";
  }
  // Not sure what it is...
  else
  {
    throw std::invalid_argument("unknown parameter type" + data.cppType);
  }
}

/**
 * Return a string representing the command-line type of a vector.
 */
template<typename T>
std::string PrintTypeDoc(
    util::ParamData& data,
    const typename std::enable_if<util::IsStdVector<T>::value>::type*)
{
  if (std::is_same<T, std::vector<int>>::value)
  {
    return "A vector of integers, separated by commas (i.e., \"1,2,3\").";
  }
  else if (std::is_same<T, std::vector<std::string>>::value)
  {
    return "A vector of strings, separated by commas (i.e., "
        "\"hello\",\"goodbye\").";
  }
  else
  {
    throw std::invalid_argument("unknown vector type" + data.cppType);
  }
}

/**
 * Return a string representing the command-line type of a matrix option.
 */
template<typename T>
std::string PrintTypeDoc(
    util::ParamData& data,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type*)
{
  if (std::is_same<T, arma::mat>::value)
  {
    return "A data matrix filename.  The file can be CSV (.csv), TSV (.csv), "
        "ASCII (space-separated values, .txt), Armadillo ASCII (.txt), PGM "
        "(.pgm), PPM (.ppm), Armadillo binary (.bin), or HDF5 (.h5, .hdf, "
        ".hdf5, or .he5), if mlpack was compiled with HDF5 support.  The type "
        "of the data is detected by the extension of the filename.  The storage"
        " should be such that one row corresponds to one point, and one column "
        "corresponds to one dimension (this is the typical storage format for "
        "on-disk data).  CSV files will be checked for a header; if no header "
        "is found, the first row will be loaded as a data point.  All values of"
        " the matrix will be loaded as double-precision floating point data.";
  }
  else if (std::is_same<T, arma::Mat<size_t>>::value)
  {
    return "A data matrix filename, where the matrix holds only non-negative "
        "integer values.  This type is often used for labels or indices.  The "
        "file can be CSV (.csv), TSV (.csv), ASCII (space-separated values, "
        ".txt), Armadillo ASCII (.txt), PGM (.pgm), PPM (.ppm), Armadillo "
        "binary (.bin), or HDF5 (.h5, .hdf, .hdf5, or .he5), if mlpack was "
        "compiled with HDF5 support.  The type of the data is detected by the "
        "extension of the filename.  The storage should be such that one row "
        "corresponds to one point, and one column corresponds to one dimension "
        "(this is the typical storage format for on-disk data).  CSV files will"
        " be checked for a header; if no header is found, the first row will be"
        " loaded as a data point.  All values of the matrix will be loaded as "
        "unsigned integers.";
  }
  else if (std::is_same<T, arma::rowvec>::value ||
           std::is_same<T, arma::vec>::value)
  {
    return "A one-dimensional vector filename.  This file can take the same "
        "formats as the data matrix filenames; however, it must either contain "
        "one row and many columns, or one column and many rows.";
  }
  else if (std::is_same<T, arma::Row<size_t>>::value ||
           std::is_same<T, arma::Col<size_t>>::value)
  {
    return "A one-dimensional vector filename, where the matrix holds only non-"
        "negative integer values.  This type is typically used for labels or "
        "predictions or other indices.  This file can take the same formats as "
        "the data matrix filenames; however, it must either contain one row and"
        " many columns, or one column and many rows.";
  }
  else
  {
    throw std::invalid_argument("unknown matrix type " + data.cppType);
  }
}

/**
 * Return a string representing the command-line type of a matrix tuple option.
 */
template<typename T>
std::string PrintTypeDoc(
    util::ParamData& /* data */,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  return "A filename for a data matrix that can contain categorical "
      "(non-numeric) data.  If the file contains only numeric data, then the "
      "same formats for regular data matrices can be used.  If the file "
      "contains strings or other values that can't be parsed as numbers, then "
      "the type to be loaded must be CSV (.csv) or ARFF (.arff).  Any non-"
      "numeric data will be converted to an unsigned integer value, and "
      "dimensions where the data is converted will be treated as categorical "
      "dimensions.  When using this format, there is no need for one-hot "
      "encoding of categorical data.";
}

/**
 * Return a string representing the command-line type of a model.
 */
template<typename T>
std::string PrintTypeDoc(
    util::ParamData& /* data */,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type*,
    const typename std::enable_if<data::HasSerialize<T>::value>::type*)
{
  return "A filename containing an mlpack model.  These can have one of three "
      "formats: binary (.bin), text (.txt), and XML (.xml).  The XML format "
      "produces the largest (but most human-readable) files, while the binary "
      "format can be significantly more compact and quicker to load and save.";
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif

/**
 * @file save_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of save functionality.
 */
#ifndef __MLPACK_CORE_DATA_SAVE_HPP
#error "Don't include this file directly; include mlpack/core/data/save.hpp."
#endif

#ifndef __MLPACK_CORE_DATA_SAVE_IMPL_HPP
#define __MLPACK_CORE_DATA_SAVE_IMPL_HPP

namespace mlpack {
namespace data {

template<typename eT>
bool Save(const std::string& filename, const arma::Mat<eT>& matrix, bool fatal)
{
  // First we will try to discriminate by file extension.
  size_t ext = filename.rfind('.');
  if (ext == std::string::npos)
  {
    if (fatal)
      Log::Fatal << "No extension given with filename '" << filename << "'; "
          << "type unknown.  Save failed." << std::endl;
    else
      Log::Warn << "No extension given with filename '" << filename << "'; "
          << "type unknown.  Save failed." << std::endl;

    return false;
  }

  // Get the actual extension.
  std::string extension = filename.substr(ext + 1);

  // Catch errors opening the file.
  std::fstream stream;
  stream.open(filename.c_str(), std::fstream::out);

  if (!stream.is_open())
  {
    if (fatal)
      Log::Fatal << "Cannot open file '" << filename << "' for writing. "
          << "Save failed." << std::endl;
    else
      Log::Warn << "Cannot open file '" << filename << "' for writing; save "
          << "failed." << std::endl;

    return false;
  }

  bool unknown_type = false;
  arma::file_type save_type;
  std::string string_type;

  if (extension == "csv")
  {
    save_type = arma::csv_ascii;
    string_type = "CSV data";
  }
  else if (extension == "txt")
  {
    save_type = arma::raw_ascii;
    string_type = "raw ASCII formatted data";
  }
  else if (extension == "bin")
  {
    save_type = arma::arma_binary;
    string_type = "Armadillo binary formatted data";
  }
  else if (extension == "pgm")
  {
    save_type = arma::pgm_binary;
    string_type = "PGM data";
  }
  else
  {
    unknown_type = true;
  }

  // Provide error if we don't know the type.
  if (unknown_type)
  {
    if (fatal)
      Log::Fatal << "Unable to determine format to save to from filename '"
          << filename << "'.  Save failed." << std::endl;
    else
      Log::Warn << "Unable to determine format to save to from filename '"
          << filename << "'.  Save failed." << std::endl;
  }

  // Try to save the file.
  Log::Info << "Saving " << string_type << " to '" << filename << "'."
      << std::endl;

  // Transpose the matrix.
  arma::Mat<eT> tmp = trans(matrix);

  if (!tmp.quiet_save(stream, save_type))
  {
    if (fatal)
      Log::Fatal << "Save to '" << filename << "' failed." << std::endl;
    else
      Log::Warn << "Save to '" << filename << "' failed." << std::endl;

    return false;
  }

  // Finally return success.
  return true;
}

}; // namespace data
}; // namespace mlpack

#endif

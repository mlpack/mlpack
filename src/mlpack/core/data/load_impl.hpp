/**
 * @file load_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of templatized load() function defined in load.hpp.
 */
#ifndef __MLPACK_CORE_DATA_LOAD_HPP
#error "Don't include this file directly; include mlpack/core/data/load.hpp."
#endif

#ifndef __MLPACK_CORE_DATA_LOAD_IMPL_HPP
#define __MLPACK_CORE_DATA_LOAD_IMPL_HPP

namespace mlpack {
namespace data {

template<typename eT>
bool Load(const std::string& filename, arma::Mat<eT>& matrix, bool fatal)
{
  // First we will try to discriminate by file extension.
  size_t ext = filename.rfind('.');
  if (ext == std::string::npos)
  {
    if (fatal)
      Log::Fatal << "Cannot determine type of file '" << filename << "'; "
          << "no extension is present." << std::endl;
    else
      Log::Warn << "Cannot determine type of file '" << filename << "'; "
          << "no extension is present.  Load failed." << std::endl;

    return false;
  }

  std::string extension = filename.substr(ext + 1);

  // Catch nonexistent files by opening the stream ourselves.
  std::fstream stream;
  stream.open(filename.c_str(), std::fstream::in);

  if (!stream.is_open())
  {
    if (fatal)
      Log::Fatal << "Cannot open file '" << filename << "'. " << std::endl;
    else
      Log::Warn << "Cannot open file '" << filename << "'; load failed."
          << std::endl;

    return false;
  }

  bool unknown_type = false;
  arma::file_type load_type;
  std::string string_type;

  if (extension == "csv")
  {
    load_type = arma::csv_ascii;
    string_type = "CSV data";
  }
  else if (extension == "txt")
  {
    // This could be raw ASCII or Armadillo ASCII (ASCII with size header).
    // We'll let Armadillo do its guessing (although we have to check if it is
    // arma_ascii ourselves) and see what we come up with.

    // This is taken from load_auto_detect() in diskio_meat.hpp
    const std::string ARMA_MAT_TXT = "ARMA_MAT_TXT";
    char* raw_header = new char[ARMA_MAT_TXT.length() + 1];
    std::streampos pos = stream.tellg();

    stream.read(raw_header, std::streamsize(ARMA_MAT_TXT.length()));
    raw_header[ARMA_MAT_TXT.length()] = '\0';
    stream.clear();
    stream.seekg(pos); // Reset stream position after peeking.

    if (std::string(raw_header) == ARMA_MAT_TXT)
    {
      load_type = arma::arma_ascii;
      string_type = "Armadillo ASCII formatted data";
    }
    else // It's not arma_ascii.  Now we let Armadillo guess.
    {
      load_type = arma::diskio::guess_file_type(stream);

      if (load_type == arma::raw_ascii) // Raw ASCII (space-separated).
        string_type = "raw ASCII formatted data";
      else if (load_type == arma::csv_ascii) // CSV can be .txt too.
        string_type = "CSV data";
      else // Unknown .txt... we will throw an error.
        unknown_type = true;
    }

    delete[] raw_header;
  }
  else if (extension == "bin")
  {
    // This could be raw binary or Armadillo binary (binary with header).  We
    // will check to see if it is Armadillo binary.
    const std::string ARMA_MAT_BIN = "ARMA_MAT_BIN";
    char *raw_header = new char[ARMA_MAT_BIN.length() + 1];

    std::streampos pos = stream.tellg();

    stream.read(raw_header, std::streamsize(ARMA_MAT_BIN.length()));
    raw_header[ARMA_MAT_BIN.length()] = '\0';
    stream.clear();
    stream.seekg(pos); // Reset stream position after peeking.

    if (std::string(raw_header) == ARMA_MAT_BIN)
    {
      string_type = "Armadillo binary formatted data";
      load_type = arma::arma_binary;
    }
    else // We can only assume it's raw binary.
    {
      string_type = "raw binary formatted data";
      load_type = arma::raw_binary;
    }

    delete[] raw_header;
  }
  else if (extension == "pgm")
  {
    load_type = arma::pgm_binary;
    string_type = "PGM data";
  }
  else // Unknown extension...
  {
    unknown_type = true;
  }

  // Provide error if we don't know the type.
  if (unknown_type)
  {
    if (fatal)
      Log::Fatal << "Unable to detect type of '" << filename << "'; "
          << "incorrect extension?" << std::endl;
    else
      Log::Warn << "Unable to detect type of '" << filename << "'; load failed."
          << " Incorrect extension?" << std::endl;

    return false;
  }

  // Try to load the file; but if it's raw_binary, it could be a problem.
  if (load_type == arma::raw_binary)
    Log::Warn << "Loading '" << filename << "' as " << string_type << "; "
        << "but this may not be the actual filetype!" << std::endl;
  else
    Log::Info << "Loading '" << filename << "' as " << string_type << "."
        << std::endl;

  bool success = matrix.load(stream, load_type);

  if (!success)
  {
    if (fatal)
      Log::Fatal << "Loading from '" << filename << "' failed." << std::endl;
    else
      Log::Warn << "Loading from '" << filename << "' failed." << std::endl;
  }

  // Now transpose the matrix.
  matrix = trans(matrix);

  // Finally, return the success indicator.
  return success;
}

}; // namespace data
}; // namespace mlpack

#endif

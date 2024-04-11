/**
 * @file core/data/detect_file_type_impl.hpp
 * @author Conrad Sanderson
 * @author Ryan Curtin
 *
 * Functionality to guess the type of a file by inspecting it.  Parts of the
 * implementation are adapted from the Armadillo sources and relicensed to be a
 * part of mlpack with permission from Conrad.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "extension.hpp"
#include "detect_file_type.hpp"
#include "string_algorithms.hpp"

namespace mlpack {
namespace data {

/**
 * Given a file type, return a logical name corresponding to that file type.
 *
 * @param type Type to get the logical name of.
 */
inline std::string GetStringType(const FileType& type)
{
  switch (type)
  {
    case FileType::CSVASCII:    return "CSV data";
    case FileType::RawASCII:    return "raw ASCII formatted data";
    case FileType::RawBinary:   return "raw binary formatted data";
    case FileType::ArmaASCII:   return "Armadillo ASCII formatted data";
    case FileType::ArmaBinary:  return "Armadillo binary formatted data";
    case FileType::PGMBinary:   return "PGM data";
    case FileType::HDF5Binary:  return "HDF5 data";
    case FileType::CoordASCII:  return "ASCII formatted sparse coordinate data";
    default:                    return "";
  }
}

/**
 * Given an istream, attempt to guess the file type.  This is taken originally
 * from Armadillo's function guess_file_type_internal(), but we avoid using
 * internal Armadillo functionality.
 *
 * If the file is detected as a CSV, and the CSV is detected to have a header
 * row, the stream `f` will be fast-forwarded to point at the second line of the
 * file.
 *
 * @param f Opened istream to look into to guess the file type.
 */
inline FileType GuessFileType(std::istream& f)
{
  f.clear();
  const std::fstream::pos_type pos1 = f.tellg();

  f.clear();
  f.seekg(0, std::ios::end);

  f.clear();
  // Get the length of the stream.
  const std::fstream::pos_type pos2 = f.tellg();

  // Compute length of the stream.
  const arma::uword nMax = ((pos1 >= 0) && (pos2 >= 0) && (pos2 > pos1)) ?
      arma::uword(pos2 - pos1) : arma::uword(0);

  f.clear();
  f.seekg(pos1);

  // Handle empty files.
  if (nMax == 0)
    return FileType::FileTypeUnknown;

  const arma::uword nUse = std::min(nMax, arma::uword(4096));

  unsigned char* dataMem = new unsigned char[nUse];
  memset(dataMem, 0, nUse);

  f.clear();
  f.read(reinterpret_cast<char*>(dataMem), std::streamsize(nUse));

  const bool loadOkay = f.good();

  f.clear();
  f.seekg(pos1);

  if (!loadOkay)
  {
    delete[] dataMem;
    return FileType::FileTypeUnknown;
  }

  bool hasBinary = false;
  bool hasBracket = false;
  bool hasComma = false;

  for (arma::uword i = 0; i < nUse; ++i)
  {
    const unsigned char val = dataMem[i];
    if ((val <= 8) || (val >= 123))
    {
      hasBinary = true;
      break;
    }  // The range checking can be made more elaborate.

    if ((val == '(') || (val == ')'))
    {
      hasBracket = true;
    }
    if (val == ',')
    {
      hasComma = true;
    }
  }

  if (hasComma && (hasBracket == false))
  {
    // If we believe we have a CSV file, then we want to try to skip any header
    // row.  We'll detect a header row by simply seeing if anything in the first
    // line doesn't parse as a number.
    //
    // TODO: this is not a foolproof algorithm, so there should eventually be a
    // way added for the user to explicitly indicate that there is or isn't a
    // header.
    std::string firstLine;
    std::getline(f, firstLine);

    std::stringstream str(firstLine);
    std::string token;
    bool allNumeric = true;
    // We'll abuse 'getline()' to split on commas.
    while (std::getline(str, token, ','))
    {
      // Let's see if we can parse the token into a number.
      double num;
      std::string rest;

      // Try to parse into a number.
      std::stringstream s(token);
      s >> num;
      if (s.fail())
      {
        allNumeric = false;
        break;
      }

      // Now check to see there isn't anything else.  (This catches cases like,
      // e.g., "1a".)
      s >> rest;
      if (rest.length() > 0)
      {
        allNumeric = false;
        break;
      }
    }

    // If we could parse everything into a number, then let's rewind `f` so that
    // it's at the start of the file.
    if (allNumeric)
      f.seekg(pos1);
  }

  delete[] dataMem;

  if (hasBinary)
    return FileType::RawBinary;

  if (hasComma && (hasBracket == false))
    return FileType::CSVASCII;

  return FileType::RawASCII;
}

/**
 * Attempt to auto-detect the type of a file given its extension, and by
 * inspecting the parts of the file to disambiguate between types when
 * necessary.  (For instance, a .csv file could be delimited by spaces, commas,
 * or tabs.)  This is meant to be used during loading.
 *
 * If the file is detected as a CSV, and the CSV is detected to have a header
 * row, `stream` will be fast-forwarded to point at the second line of the file.
 *
 * @param stream Opened file stream to look into for autodetection.
 * @param filename Name of the file.
 * @return The detected file type.
 */
inline FileType AutoDetect(std::fstream& stream, const std::string& filename)
{
  // Get the extension.
  std::string extension = Extension(filename);
  FileType detectedLoadType = FileType::FileTypeUnknown;

  if (extension == "csv" || extension == "tsv")
  {
    detectedLoadType = GuessFileType(stream);
    if (detectedLoadType == FileType::CSVASCII)
    {
      if (extension == "tsv")
        Log::Warn << "'" << filename << "' is comma-separated, not "
            "tab-separated!" << std::endl;
    }
    else if (detectedLoadType == FileType::RawASCII) // .csv file can be tsv.
    {
      if (extension == "csv")
      {
        // We should issue a warning, but we don't want to issue the warning if
        // there is only one column in the CSV (since there will be no commas
        // anyway, and it will be detected as arma::raw_ascii).
        const std::streampos pos = stream.tellg();
        std::string line;
        std::getline(stream, line, '\n');
        Trim(line);

        // Reset stream position.
        stream.seekg(pos);

        // If there are no spaces or whitespace in the line, then we shouldn't
        // print the warning.
        if ((line.find(' ') != std::string::npos) ||
            (line.find('\t') != std::string::npos))
        {
          Log::Warn << "'" << filename << "' is not a standard csv file."
              << std::endl;
        }
      }
    }
    else
    {
      detectedLoadType = FileType::FileTypeUnknown;
    }
  }
  else if (extension == "txt")
  {
    // This could be raw ASCII or Armadillo ASCII (ASCII with size header).
    // We'll let Armadillo do its guessing (although we have to check if it is
    // arma_ascii ourselves) and see what we come up with.

    // This is adapted from load_auto_detect() in diskio_meat.hpp.
    const std::string ARMA_MAT_TXT = "ARMA_MAT_TXT";
    std::string rawHeader(ARMA_MAT_TXT.length(), '\0');
    std::streampos pos = stream.tellg();

    stream.read(&rawHeader[0], std::streamsize(ARMA_MAT_TXT.length()));
    stream.clear();
    stream.seekg(pos); // Reset stream position after peeking.

    if (rawHeader == ARMA_MAT_TXT)
    {
      detectedLoadType = FileType::ArmaASCII;
    }
    else // It's not arma_ascii.  Now we let Armadillo guess.
    {
      detectedLoadType = GuessFileType(stream);

      if (detectedLoadType != FileType::RawASCII &&
          detectedLoadType != FileType::CSVASCII)
        detectedLoadType = FileType::FileTypeUnknown;
    }
  }
  else if (extension == "bin")
  {
    // This could be raw binary or Armadillo binary (binary with header).  We
    // will check to see if it is Armadillo binary.
    const std::string ARMA_MAT_BIN = "ARMA_MAT_BIN";
    const std::string ARMA_SPM_BIN = "ARMA_SPM_BIN";
    std::string rawHeader(ARMA_MAT_BIN.length(), '\0');

    std::streampos pos = stream.tellg();

    stream.read(&rawHeader[0], std::streamsize(ARMA_MAT_BIN.length()));
    stream.clear();
    stream.seekg(pos); // Reset stream position after peeking.

    if (rawHeader == ARMA_MAT_BIN || rawHeader == ARMA_SPM_BIN)
    {
      detectedLoadType = FileType::ArmaBinary;
    }
    else // We can only assume it's raw binary.
    {
      detectedLoadType = FileType::RawBinary;
    }
  }
  else if (extension == "pgm")
  {
    detectedLoadType = FileType::PGMBinary;
  }
  else if (extension == "h5" || extension == "hdf5" || extension == "hdf" ||
           extension == "he5")
  {
    detectedLoadType = FileType::HDF5Binary;
  }
  else // Unknown extension...
  {
    detectedLoadType = FileType::FileTypeUnknown;
  }

  return detectedLoadType;
}

/**
 * Return the type based only on the extension.
 *
 * @param filename Name of the file whose type we should detect.
 * @return Detected type of file.
 */
inline FileType DetectFromExtension(const std::string& filename)
{
  const std::string extension = Extension(filename);

  if (extension == "csv")
  {
    return FileType::CSVASCII;
  }
  else if (extension == "txt")
  {
    return FileType::RawASCII;
  }
  else if (extension == "bin")
  {
    return FileType::ArmaBinary;
  }
  else if (extension == "pgm")
  {
    return FileType::PGMBinary;
  }
  else if (extension == "h5" || extension == "hdf5" || extension == "hdf" ||
           extension == "he5")
  {
    return FileType::HDF5Binary;
  }
  else
  {
    return FileType::FileTypeUnknown;
  }
}

/**
 * Count the number of columns in the file.  The file must be a CSV/TSV/TXT file
 * with no header.
 */
inline size_t CountCols(std::fstream& f)
{
  f.clear();
  const std::fstream::pos_type pos1 = f.tellg();

  std::string firstLine;
  std::getline(f, firstLine);

  // Extract tokens from the first line using whitespace.
  std::stringstream str(firstLine);
  size_t cols = 0;
  std::string token;

  while (str >> token)
    ++cols;

  // Reset to wherever we were.
  f.clear();
  f.seekg(pos1);

  return cols;
}

} // namespace data
} // namespace mlpack

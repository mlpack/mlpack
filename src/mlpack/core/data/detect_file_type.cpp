/**
 * @file core/data/detect_file_type.cpp
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

#include <boost/algorithm/string/trim.hpp>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

namespace mlpack {
namespace data {

/**
 * Given a file type, return a logical name corresponding to that file type.
 *
 * @param type Type to get the logical name of.
 */
std::string GetStringType(const arma::file_type& type)
{
  switch (type)
  {
    case arma::csv_ascii:   return "CSV data";
    case arma::raw_ascii:   return "raw ASCII formatted data";
    case arma::raw_binary:  return "raw binary formatted data";
    case arma::arma_ascii:  return "Armadillo ASCII formatted data";
    case arma::arma_binary: return "Armadillo binary formatted data";
    case arma::pgm_binary:  return "PGM data";
    case arma::hdf5_binary: return "HDF5 data";
    default:                return "";
  }
}

/**
 * Given an istream, attempt to guess the file type.  This is taken originally
 * from Armadillo's function guess_file_type_internal(), but we avoid using
 * internal Armadillo functionality.
 *
 * @param f Opened istream to look into to guess the file type.
 */
arma::file_type GuessFileType(std::istream& f)
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
    return arma::file_type_unknown;

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
    return arma::file_type_unknown;
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

  delete[] dataMem;

  if (hasBinary)
    return arma::raw_binary;

  if (hasComma && (hasBracket == false))
    return arma::csv_ascii;

  return arma::raw_ascii;
}

/**
 * Attempt to auto-detect the type of a file given its extension, and by
 * inspecting the parts of the file to disambiguate between types when
 * necessary.  (For instance, a .csv file could be delimited by spaces, commas,
 * or tabs.)  This is meant to be used during loading.
 *
 * @param stream Opened file stream to look into for autodetection.
 * @param filename Name of the file.
 * @return The detected file type.
 */
arma::file_type AutoDetect(std::fstream& stream,
                           const std::string& filename)
{
  // Get the extension.
  std::string extension = Extension(filename);
  arma::file_type detectedLoadType = arma::file_type_unknown;

  if (extension == "csv" || extension == "tsv")
  {
    detectedLoadType = GuessFileType(stream);
    if (detectedLoadType == arma::csv_ascii)
    {
      if (extension == "tsv")
        Log::Warn << "'" << filename << "' is comma-separated, not "
            "tab-separated!" << std::endl;
    }
    else if (detectedLoadType == arma::raw_ascii) // .csv file can be tsv.
    {
      if (extension == "csv")
      {
        // We should issue a warning, but we don't want to issue the warning if
        // there is only one column in the CSV (since there will be no commas
        // anyway, and it will be detected as arma::raw_ascii).
        const std::streampos pos = stream.tellg();
        std::string line;
        std::getline(stream, line, '\n');
        boost::trim(line);

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
      detectedLoadType = arma::file_type_unknown;
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
      detectedLoadType = arma::arma_ascii;
    }
    else // It's not arma_ascii.  Now we let Armadillo guess.
    {
      detectedLoadType = GuessFileType(stream);

      if (detectedLoadType != arma::raw_ascii &&
          detectedLoadType != arma::csv_ascii)
        detectedLoadType = arma::file_type_unknown;
    }
  }
  else if (extension == "bin")
  {
    // This could be raw binary or Armadillo binary (binary with header).  We
    // will check to see if it is Armadillo binary.
    const std::string ARMA_MAT_BIN = "ARMA_MAT_BIN";
    std::string rawHeader(ARMA_MAT_BIN.length(), '\0');

    std::streampos pos = stream.tellg();

    stream.read(&rawHeader[0], std::streamsize(ARMA_MAT_BIN.length()));
    stream.clear();
    stream.seekg(pos); // Reset stream position after peeking.

    if (rawHeader == ARMA_MAT_BIN)
    {
      detectedLoadType = arma::arma_binary;
    }
    else // We can only assume it's raw binary.
    {
      detectedLoadType = arma::raw_binary;
    }
  }
  else if (extension == "pgm")
  {
    detectedLoadType = arma::pgm_binary;
  }
  else if (extension == "h5" || extension == "hdf5" || extension == "hdf" ||
           extension == "he5")
  {
    detectedLoadType = arma::hdf5_binary;
  }
  else // Unknown extension...
  {
    detectedLoadType = arma::file_type_unknown;
  }

  return detectedLoadType;
}

/**
 * Return the type based only on the extension.
 *
 * @param filename Name of the file whose type we should detect.
 * @return Detected type of file.
 */
arma::file_type DetectFromExtension(const std::string& filename)
{
  const std::string extension = Extension(filename);

  if (extension == "csv")
  {
    return arma::csv_ascii;
  }
  else if (extension == "txt")
  {
    return arma::raw_ascii;
  }
  else if (extension == "bin")
  {
    return arma::arma_binary;
  }
  else if (extension == "pgm")
  {
    return arma::pgm_binary;
  }
  else if (extension == "h5" || extension == "hdf5" || extension == "hdf" ||
           extension == "he5")
  {
    return arma::hdf5_binary;
  }
  else
  {
    return arma::file_type_unknown;
  }
}

} // namespace data
} // namespace mlpack

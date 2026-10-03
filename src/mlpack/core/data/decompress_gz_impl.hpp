/**
 * @file core/data/decompress_gz_impl.hpp
 * @author Omar Shrit
 *
 * Implementation of standalone gzip decompression using zlib.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_DECOMPRESS_GZ_IMPL_HPP
#define MLPACK_CORE_DATA_DECOMPRESS_GZ_IMPL_HPP

#include "decompress_gz.hpp"
#include "extension.hpp"
#include "handle_files.hpp"

#ifdef MLPACK_USE_ZLIB
  #include <zlib.h>
#endif

namespace mlpack {

inline bool IsGzipFile(const std::string& filePath)
{
  std::ifstream f(filePath, std::ios::binary);
  if (!f)
    return false;

  unsigned char magic[2] = {0, 0};
  f.read(reinterpret_cast<char*>(magic), 2);

  return f.good() && magic[0] == 0x1F && magic[1] == 0x8B;
}

#ifdef MLPACK_USE_ZLIB

inline void DecompressGzFile(const std::string& srcPath,
                             const std::string& destPath)
{
  gzFile gzIn = gzopen(srcPath.c_str(), "rb");
  if (!gzIn)
  {
    throw std::runtime_error("DecompressGzFile(): cannot open '" + srcPath
        + "' for reading.");
  }

  std::ofstream out(destPath, std::ios::binary);
  if (!out)
  {
    gzclose(gzIn);
    throw std::runtime_error("DecompressGzFile(): cannot open '" + destPath
        + "' for writing.");
  }

  constexpr size_t bufSize = 131072; // 128 KB buffer.
  std::vector<char> buf(bufSize);

  int bytesRead = 0;
  while ((bytesRead = gzread(gzIn, buf.data(), bufSize)) > 0)
  {
    out.write(buf.data(), bytesRead);
    if (!out.good())
    {
      gzclose(gzIn);
      throw std::runtime_error("DecompressGzFile(): error writing to '"
          + destPath + "'.");
    }
  }

  if (bytesRead < 0)
  {
    int errnum = 0;
    const char* errMsg = gzerror(gzIn, &errnum);
    std::string msg(errMsg ? errMsg : "unknown error");
    gzclose(gzIn);
    throw std::runtime_error("DecompressGzFile(): zlib error while "
        "decompressing '" + srcPath + "': " + msg);
  }

  gzclose(gzIn);
}

#endif // MLPACK_USE_ZLIB

inline void DecompressGzipIfNeeded(std::string& filename)
{
  std::string ext = Extension(filename);
  if (ext != "gz")
    return;

#ifdef MLPACK_USE_ZLIB
  if (!IsGzipFile(filename))
    return;

  // Extract the underlying extension in order to get the temp file name.
  std::string innerName = filename.substr(0, filename.size() - 3);
  std::string innerExt = Extension(innerName);

  std::filesystem::path dest = TempName();
  if (!innerExt.empty())
    dest += "." + innerExt;

  DecompressGzFile(filename, dest.string());
  filename = dest.string();

#else
  throw std::runtime_error("Load(): the file '" + filename + "' is "
      "gzip-compressed but MLPACK_USE_ZLIB is not enabled.  Enable zlib "
      "support by adding '#define MLPACK_USE_ZLIB' before including mlpack "
      "and linking with -lz.");
#endif
}

} // namespace mlpack

#endif

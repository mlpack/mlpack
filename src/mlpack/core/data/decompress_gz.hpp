/**
 * @file core/data/decompress_gz.hpp
 * @author Omar Shrit
 *
 * gzip decompression using zlib.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_DECOMPRESS_GZ_HPP
#define MLPACK_CORE_DATA_DECOMPRESS_GZ_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Check if a file contains gzip-compressed data by inspecting the first two
 * bytes for the gzip magic number (0x1F 0x8B).
 *
 * @param filePath Path to the file to check.
 * @return true if the file starts with the gzip magic bytes.
 */
inline bool IsGzipFile(const std::string& filePath);

#ifdef MLPACK_USE_ZLIB

/**
 * Decompress a gzip file and write the result to a new file.  Uses zlib's
 * gzopen/gzread API directly.
 *
 * @param srcPath Path to the gzip-compressed file.
 * @param destPath Path where the decompressed file will be written.
 */
inline void DecompressGzFile(const std::string& srcPath,
                             const std::string& destPath);

#endif // MLPACK_USE_ZLIB

/**
 * If `filename` has a .gz extension, decompress it to a temporary file and
 * update `filename` to point to the decompressed result.  The original .gz
 * extension is stripped and replaced with the underlying extension (e.g.
 * "data.csv.gz" produces a temp file ending in ".csv").
 *
 * If the file is not gzip-compressed (no .gz extension, or the magic bytes
 * don't match), `filename` is left unchanged.
 *
 * @param filename On input, path to the (possibly compressed) file.  On
 *     output, path to the decompressed file (or unchanged if no decompression
 *     was needed).
 */
inline void DecompressGzipIfNeeded(std::string& filename);

} // namespace mlpack

#include "decompress_gz_impl.hpp"

#endif

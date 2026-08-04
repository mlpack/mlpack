/**
 * @file core/data/download_file.hpp
 * @author Omar Shrit
 *
 * Functions to download datasets from a URL, with optional caching.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_DOWNLOAD_FILE_HPP
#define MLPACK_CORE_DATA_DOWNLOAD_FILE_HPP

// In case if it is not included
#include "../httplib/httplib.hpp"

#include <fstream>
#include <sstream>
#include <unordered_map>

namespace mlpack {

/*
 * Check if the provided string is a valid URL (starts with http:// or https://).
 *
 * @param url String to be checked.
 * @return true if the string is a URL, false otherwise.
 */
inline bool CheckIfURL(const std::string& url);

/*
 * Check if the provided hostname contains only valid characters (lowercase
 * letters, digits, hyphens and dots).
 *
 * @param host Hostname to validate.
 * @return true if the hostname is valid.
 */
inline bool CheckValidHost(const std::string& host);

/*
 * Check if a string is composed entirely of digits.
 *
 * @param str String to check.
 * @return true if every character in str is a digit.
 */
inline bool IsDigits(const std::string& str);

/*
 * Parse a given URL and try to extract hostname, filename and the port
 * number.
 *
 * @param url Given URL to parse.
 * @param host Extracted hostname; throws on failure.
 * @param filename Extracted filename from the URL path.
 * @param port Extracted port number, or -1 if not specified.
 */
inline void ParseURL(const std::string& url, std::string& host,
                     std::string& filename, int& port);

// The cache manifest maps a URL to (etag, content_length, local_path).
using CacheManifest =
    std::unordered_map<std::string,
                       std::tuple<std::string, size_t, std::string>>;

/*
 * The idea is to allow three cache pattern, the first one, user defined
 * macro, the second one a default cache path if no user defined, and the last
 * one is the system temp directory.
 *
 * Following follows Bandicoot's kernel-cache pattern, defined in cache_meat.hpp
 *   1. MLPACK_CACHE_DIR (compile-time macro)
 *   2. $HOME/.mlpack/cache/  (POSIX) or %APPDATA%\mlpack\cache\ (Windows)
 *   3. system temp directory
 *
 * @return The absolute path to the cache directory.
 */
inline std::string GetCacheDir();

/*
 * Load the cache manifest (cache_manifest.csv) from disk. This allow to use
 * previous cached dataset from previous sessions
 *
 * @param cacheDir Path to the cache directory.
 * @param manifest Filled with the cached entries on success.
 * @return true if the manifest was read, false if it does not exist yet.
 */
inline bool LoadManifest(const std::string& cacheDir,
                         CacheManifest& manifest);

/*
 * Write the cache manifest back to disk, overwriting the previous one.
 *
 * @param cacheDir Path to the cache directory.
 * @param manifest The manifest to write.
 * @return true on success, false on failure.
 */
inline bool SaveManifest(const std::string& cacheDir,
                         const CacheManifest& manifest);

/*
 * Download a file from a URL and save it to a user-specified destination.
 * Checks the cache first, if the ETag and size match and dest already exists,
 * the download is skipped.
 *
 * @param url Given URL to download dataset from.
 * @param dest The local path to save the downloaded file to.
 * @return true on success, throws on failure.
 */
inline bool DownloadFile(const std::string& url,
                         const std::string& dest);

/*
 * Internal function used by Load() to download a file from a URL.
 * The file is stored in the cache directory.  When caching is disabled,
 * the file is saved to a generated temporary path.
 * Either way, `filename` is set to the local path where the data ended up.
 *
 * @param url Given URL to download dataset from.
 * @param filename Set to the local path of the downloaded (or cached) file.
 * @return true if download or cache is successful, otherwise throws.
 */
inline bool DownloadFileInternal(const std::string& url,
                                 std::string& filename);

} // namespace mlpack

#include "download_file_impl.hpp"

#endif

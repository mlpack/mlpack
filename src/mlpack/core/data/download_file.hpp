/**
 * @file core/data/download_file.hpp
 * @author Omar Shrit
 *
 * mlpack Load function that download dataset from a URL.
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

namespace mlpack {

/*
 * Check if the provided URL is valid or not.
 *
 * @param url to be checked.
 * @return false on failure.
 */
inline bool CheckIfURL(const std::string& url);

/*
 * Parse a given URL and try to extract hostname, filename and the port
 * number.
 *
 * @param url Given URL to download dataset from.
 * @param Extract hostname, throw exception on failure
 * @param filename Try to extract the filename of the downloaded file.
 * @param port Try To extract the port number from the url if provided.
 * @return void, only throws exception on failure
 */
inline void ParseURL(const std::string& url, std::string& host,
                     std::string& filename, int& port);

/*
 * Try to download a file from a URL provided by the user.
 *
 * @param url Given URL to download dataset from.
 * @param filename The path to save the downloaded file to; if empty,
 * then a temporary filename is generated.
 * @return true if download is successful, otherwise, throw error on failure, or
 * return false.
 */
inline bool DownloadFile(const std::string& url,
                         std::string& filename);

} // namespace mlpack

#include "download_file_impl.hpp"

#endif

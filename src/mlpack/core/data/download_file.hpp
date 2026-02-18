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

#ifdef MLPACK_ENABLE_HTTPLIB

namespace mlpack {

/**
 * return a true if the URL is provided.
 */
inline bool CheckIfURL(const std::string& url)
{
  std::regex rgx("^https?://");
  std::smatch match;
  if (std::regex_search(url, match, rgx))
  {
    return true;
  }
  return false;
}

/**
 * Given an URL, extract the filename that is being downloaded using regex.
 *
 * @param filename to be extracted from URL.
 * @param url that contains the filename at the end.
 */
inline void FilenameFromURL(std::string& filename, const std::string& url)
{
  std::regex rgx("[^/]+(?=/$|$)");
  std::smatch match;
  if (std::regex_search(url, match, rgx))
  {
    //std::cout << "filename: " << match[0] << std::endl;
    filename = match[0];
  }
}

/**
 * Extract host from URL.
 */
inline std::string URLToHost(const std::string& url)
{
  std::string host;
  std::regex rgx(R"(^(?:https?|ftp)://(?:[^@/\n]+@)?([^:/?\n]+))");
  std::smatch match;
  if (std::regex_search(url, match, rgx))
  {
    host = match[1];
  }
  return host;
}

bool DownloadFile(const std::string& url,
                  std::string& filename)
{
  std::fstream stream;
  // If host is not extracted correctly, we will get a segmentation fault from
  // httplib
  std::string host = URLToHost(url);
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  httplib::SSLClient cli(host, 443);
#else
  auto port = 80;
  httplib::Client cli(host, port);
#endif
  cli.set_connection_timeout(2);
  httplib::Result res = cli.Get(url);

  if (res->status != 200)
  {
    std::stringstream oss;
    oss <<  "Unable to connect, status returned: '" << res->status;
    throw std::runtime_error(oss.str());
  }

  std::string originalFilename;
  FilenameFromURL(originalFilename, url);

  filename = TempName();
  // This is necessary to get the extension.
  filename += originalFilename;
  // @rcurtin, I do not like this, but this is the only option;
  DataOptions opts = NoFatal;
  if (!OpenFile(filename, opts, false, stream))
  {
    std::stringstream oss;
    oss <<  "Unable to open a temporary file for downloading data.";
    throw std::runtime_error(oss.str());
  }

  stream.write(res->body.data(), res->body.size());
  if (!stream.good())
  {
    std::stringstream oss;
    oss << "Error writing to a '" << filename << "'.  "
          << "Please check permissions or disk space.";
    throw std::runtime_error(oss.str());
  }
  stream.close();
  return true;
}

} // namespace mlpack

#endif

#endif

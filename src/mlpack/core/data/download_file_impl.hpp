/**
 * @file core/data/download_file_impl.hpp
 * @author Omar Shrit
 *
 * mlpack Load function that download dataset from a URL.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_DOWNLOAD_FILE_IMPL_HPP
#define MLPACK_CORE_DATA_DOWNLOAD_FILE_IMPL_HPP

#ifdef MLPACK_ENABLE_HTTPLIB

#include "download_file.hpp"

namespace mlpack {

inline bool CheckIfURL(const std::string& url)
{
  if (!url.empty())
  {
    if (url.compare(0, 7, "http://") == 0 ||
        url.compare(0, 8, "https://") == 0)
    {
      return true;
    }
  }
  return false;
}

inline void ParseURL(const std::string& url, std::string& host,
                     std::string& filename, int& port)
{
  if (!CheckIfURL(url) || url.size() <= 8)
  {
    throw std::runtime_error("Invalid URL provided."
        " URL should start with http or https");
  }
  size_t pos = url.find("://");
  pos = pos + 3;

  std::string possibleHost = url.substr(pos);

  size_t hostPos = possibleHost.find_first_of(".:/");
  if (hostPos == std::string::npos)
  {
    throw std::runtime_error("Domain name is not valid."
        " Domain name should contains '.' between the hostname and the top"
        " level domain. Or '/' at the end. Please check the provided URL");
  }

  size_t endHost = possibleHost.find_first_of(":/");
  if (endHost != std::string::npos)
  {
    host = possibleHost.substr(0, endHost);
    char endChar = possibleHost.at(endHost);
    if (endChar == ':')
    {
      // We need to find the last char which is /
      std::string findPort = possibleHost.substr(endHost + 1);
      size_t endPort = findPort.find("/");
      port = std::stoi(findPort.substr(0, endPort));
    }
  }

  int firstPos = possibleHost.rfind(":");
  int secPos   = possibleHost.rfind("/");
  // Need to be sure that we are comparing valid number since npos is the
  // highest possible value in size_t
  if (firstPos == (int)std::string::npos) firstPos = -1;
  if (secPos == (int)std::string::npos) secPos = -1;
  if (secPos > firstPos)
  {
    size_t filePos = possibleHost.rfind("/");
    // no need to throw an exception, if the file is not found this is not a
    // problem with the URL.
    if (filePos != std::string::npos)
    {
      std::string possibleFilename = possibleHost.substr(filePos + 1);
      size_t posFile = possibleFilename.find_first_of("?#");
      // Check for the '.' as a marker for extension
      if (possibleFilename.find(".") != std::string::npos)
      {
        // we assume something is after the file name.
        if (posFile != std::string::npos)
        {
          filename = possibleFilename.substr(0, posFile);
        }
        else
        {
          filename = possibleFilename;
        }
      }
    }
  }
}

inline bool DownloadFile(const std::string& url,
                         std::string& filename)
{
  std::fstream stream;
  int port = -1;
  std::string host;
  ParseURL(url, host, filename, port);

  // Sanity check if in case.
  if (host.empty())
  {
    throw std::runtime_error("Domain name could not be parsed.");
  }
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  httplib::SSLClient cli(host, 443);
#else
  if (port == -1)
  {
    port = 80;
  }
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

  std::filesystem::path tmpFilename = TempName();
  // This is necessary to get the extension.
  tmpFilename += filename;
  // @rcurtin, I do not like this, but this is the only option;
  // Or I can take the internal of OpenFile, and use it here.
  DataOptions opts = NoFatal;
  if (!OpenFile(filename, opts, false, stream))
  {
    std::stringstream oss;
    oss <<  "Unable to open a temporary file for downloading data.";
    throw std::runtime_error(oss.str());
  }
  errno = 0;
  stream.write(res->body.data(), res->body.size());
  stream.flush();
  if (!stream.good())
  {
    std::stringstream oss;
    oss << "Error writing to a '" << tmpFilename << "' failed: "
        << std::strerror(errno);
    throw std::runtime_error(oss.str());
  }
  stream.close();
  return true;
}

} // namespace mlpack

#endif

#endif

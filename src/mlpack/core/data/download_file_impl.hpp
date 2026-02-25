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

#include "download_file.hpp"

namespace mlpack {

#ifdef MLPACK_ENABLE_HTTPLIB

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

inline bool CheckValidHost(const std::string& host)
{
  if (host.empty())
    return false;
  else if (host.size() > 253)
    return false;

  std::string traillingDots("..");
  if (host.find(traillingDots) != std::string::npos)
    return false;

  // Only lower case alpha, numeric and hyphen are allowed in a hostname.
  struct InvalidHost
  {
    bool operator()(char c)
    {
      if (std::isdigit(c))
        return false;
      else if (std::islower(c))
        return false;
      else if (c == '-' || c== '.')
        return false;
      else
        return true;
    }
  };
  return std::find_if(host.begin(), host.end(), InvalidHost()) == host.end();
}

inline bool IsDigits(const std::string &str)
{
  return std::all_of(str.begin(), str.end(), ::isdigit);
}

inline void ParseURL(const std::string& url, std::string& host,
                     std::string& filename, int& port)
{
  if (!CheckIfURL(url) || url.size() <= 8)
  {
    throw std::runtime_error("Invalid URL provided."
        " URL should start with 'http://' or 'https://'.");
  }
  size_t pos = url.find("://");
  pos = pos + 3;

  std::string possibleHost = url.substr(pos);

  //size_t hostPos = possibleHost.find_first_of(".:/");
  //if (hostPos == std::string::npos)
  //{
    //throw std::runtime_error("Domain name is not valid."
        //" Domain name should contains '.' between the hostname and the top"
        //" level domain. Or '/' at the end. Please check the provided URL");
  //}

  size_t endHost = possibleHost.find_first_of(":/");
  if (endHost != std::string::npos)
  {
    std::string tmpHost = possibleHost.substr(0, endHost);
    if (CheckValidHost(tmpHost))
      host = tmpHost;
    else
      throw std::runtime_error("Invalid Host.\n"
          "Valid host contains only lower letters, numeric and hyphen");

    char endChar = possibleHost.at(endHost);
    if (endChar == ':')
    {
      // We need to find the last char which is /
      std::string findPort = possibleHost.substr(endHost + 1);
      size_t endPort = findPort.find("/");
      if (IsDigits(findPort.substr(0, endPort)))
        port = std::stoi(findPort.substr(0, endPort));
    }
  }
  else // In case of using http://localhost or similar
  {
    if (CheckValidHost(possibleHost))
      host = possibleHost;
    else
      throw std::runtime_error("Invalid Host.\n"
          "Valid host contains only lower letters, numeric and hyphen");
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
    throw std::runtime_error("DownloadFile(): domain name could not be parsed"
        " from URL '" + url + "'");
  }
#ifdef MLPACK_USE_HTTPS
  if (port == -1)
  {
    port = 443;
  }
  httplib::SSLClient cli(host, port);
#else
  if (port == -1)
  {
    port = 80;
  }
  httplib::Client cli(host, port);
#endif
  cli.set_connection_timeout(2);
  httplib::Result res = cli.Get(url);
  if (!res)
  {
    std::stringstream oss;
    oss << "DownloadFile(): httplib error: " << httplib::to_string(res.error());
    throw std::runtime_error(oss.str());
  }

  std::filesystem::path tmpFilename = TempName();
  // This is necessary to get the extension.
  tmpFilename = tmpFilename + "." + Extension(filename);

#ifdef  _WIN32 // Always open in binary mode on Windows.
    stream.open(filename.c_str(), std::fstream::in
        | std::fstream::binary);
#else
    stream.open(filename.c_str(), std::fstream::in);
#endif

  if (!stream.is_open())
  {
    std::stringstream oss;
    oss <<  "DownloadFile(): cannot open temporary file '" << tmpFilename
        << "' for storing downloaded data. Please check the file path.";
    throw std::runtime_error(oss.str());
  }
  errno = 0;
  stream.write(res->body.data(), res->body.size());
  stream.flush();
  if (!stream.good())
  {
    std::stringstream oss;
    oss << "Error writing to file '" << tmpFilename << "': "
        << std::strerror(errno);
    throw std::runtime_error(oss.str());
  }
  stream.close();
  return true;
}

#else

/**
 * Implementation to be removed at the end of the integration.
 */
inline bool CheckIfURL(const std::string& url)
{
  std::string url2 = url; // Avoid compiler warning.
  return false;
}

inline void ParseURL(const std::string& url, std::string& host,
                     std::string& filename, int& port)
{
  host = url;
  filename = "";
  port = -1;
  std::stringstream oss;
  oss << "Cannot check the provided URL: " << url << std::endl
      << "httplib has not been enabled during compilation time." << std::endl
      << "Please enable httplib by defining this in your code:" << std::endl
      << "#define MLPACK_ENABLE_HTTPLIB" << std::endl
      << "If you would like to enable httplib when installing mlpack."
      << " Please refer to our documentation page.";
  throw std::runtime_error(oss.str());
}

inline bool DownloadFile(const std::string& url,
                         std::string& filename)
{
  filename = "";
```suggestion
  throw std::runtime_error("DownloadFile(): httplib support not enabled; cannot"
      " check provided URL '" + url + "'.  Enable httplib by adding '#define "
      "MLPACK_ENABLE_HTTPLIB' before including mlpack.");
  return false;
}

#endif

} // namespace mlpack

#endif

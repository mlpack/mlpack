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
  if (url.compare(0, 7, "http://") == 0 || url.compare(0, 8, "https://") == 0)
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
//inline bool URLToHost(const std::string& url, std::string& host)
//{
  //bool success = false;
  //std::regex rgx(R"(^(?:https?)://(?:[^@/\n]+@)?([^:/?\n]+))");
  //std::smatch match;
  //if (std::regex_search(url, match, rgx))
  //{
    //host = match[1];
  //}
  //return host;
//}


void ParseURL(const std::string& url, std::string& host,
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
  std::cout << "possible Host: " << possibleHost << std::endl;

  size_t hostPos = possibleHost.find_first_of(":/");
  if (hostPos == std::string::npos)
  {
    throw std::runtime_error("Host name is not found."
        " Host name ends either by '/' or ':'. Please check the provided URL");
  }

  host = possibleHost.substr(0, hostPos);
 // we should not use hostPos in here. 
  char endChar = possibleHost.at(hostPos);
  std::cout << "End Char:" << endChar << std::endl;
  if (endChar == ':')
  {
    // we need to find the last char which is /
    std::string findPort = possibleHost.substr(hostPos + 1);
    std::cout << "possible port: " << findPort << std::endl;
    size_t endPort = findPort.find("/");
    port = std::stoi(findPort.substr(0, endPort));
  }

  size_t filePos = url.rfind("/");
  // no need to throw an exception, if the file is not found this is not a
  // problem with the URL.
  if (filePos != std::string::npos)
  {
    std::string possibleFilename = url.substr(filePos);
    std::cout << "possible filename:" << possibleFilename << std::endl;
    size_t posFile = possibleFilename.find_first_of("?#");
    if (posFile != std::string::npos)
    {
      filename = possibleFilename.substr(posFile);
    }
  }
}

bool DownloadFile(const std::string& url,
                  std::string& filename)
{
  std::fstream stream;
  int port = -1;
  std::string host;
  ParseURL(url, host, filename, port);

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  httplib::SSLClient cli(host, 443);
#else
  if (port = -1)
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
  errno = 0;
  stream.write(res->body.data(), res->body.size());
  stream.flush();
  if (!stream.good())
  {
    std::stringstream oss;
    oss << "Error writing to a '" << filename << "' failed: "
        << std::strerror(errno);
    throw std::runtime_error(oss.str());
  }
  stream.close();
  return true;
}

} // namespace mlpack

#endif

#endif

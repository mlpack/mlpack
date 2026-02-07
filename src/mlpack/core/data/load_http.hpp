/**
 * @file core/data/load_http.hpp
 * @author Omar Shrit
 *
 * mlpack Load function that download dataset from a URL.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_HTTP_HPP
#define MLPACK_CORE_DATA_LOAD_HTTP_HPP

#ifdef MLPACK_ENABLE_HTTPLIB

namespace mlpack {

void FilenameFromURL(std::string& filename, const std::string& url)
{
  std::regex rgx("[^/]+(?=/$|$)");
  std::smatch match;
  if (std::regex_search(url, match, rgx))
  {
    //std::cout << "filename: " << match[0] << std::endl;
    filename = match[0];
  }
}

bool checkIfURL(const std::string& url)
{
  std::regex rgx("^https?://");
  std::smatch match;
  if (std::regex_search(url, match, rgx))
  {
    return true;
  }
  return false;
}

template<typename MatType>
bool DownloadFile(const std::string& url,
                  std::fstream& stream, // will see if we will need this
                  TextOptions& opts)
{
  bool success = false;

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  auto port = 443;
  httplib::SSLClient cli(url, port);
#else
  auto port = 80;
  Client cli(url, port);
#endif
  cli.set_connection_timeout(2);
  Result res = cli.Get(url);
  std::cout << "Status: " << res->status << std::endl;

  if (res->status != 200)
  {
    std::stringstream oss;
    oss <<  "Unable to connect, status returned: '" << res->status;
    return HandleError(oss, opts);
  }

  //std::cout << "Show the body of the message:"  << std::endl << res->body
   //   << std::endl;
  std::stringstream data(res->body);

//#ifdef MLPACK_CAHCHE
  FilenameFromURL(filename, url);
  success = WriteToFile(filename, opts, data.str(), stream);
//#endif 

  std::string tmpName = std::tmpnam(nullptr);
  // Maybe use WriteToFile() in here?? not really sure.
  success = OpenFile(tmpName, opts, false, stream);
  if (!success)
  {
    std::stringstream oss;
    oss <<  "Unable to open a temporary file for downloading data.";
    return HandleError(oss, opts);
  }
   
  stream.write(data.data(), data.size());
  if (!stream.good())
  {
    std::stringstream oss;
    oss << "Error writing to a '" << filename << "'.  "
          << "Please check permissions or disk space.";
    return HandleError(oss, opts);
  }
  stream.close();
  return success;
}

} // namespace mlpack

#endif

#endif

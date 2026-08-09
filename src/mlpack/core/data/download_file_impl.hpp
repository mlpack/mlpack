/**
 * @file core/data/download_file_impl.hpp
 * @author Omar Shrit
 *
 * Implementation of DownloadFile() and the URL download cache.
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

inline bool IsDigits(const std::string& str)
{
  return std::all_of(str.begin(), str.end(), ::isdigit);
}

inline void ParseURL(const std::string& url, std::string& host,
                     std::string& filename, int& port)
{
  if (!CheckIfURL(url) || url.size() <= 8)
  {
    throw std::runtime_error("ParseURL(): invalid URL '" + url + "' provided."
        " URL should start with 'http://' or 'https://'.");
  }
  size_t pos = url.find("://");
  pos = pos + 3;

  std::string possibleHost = url.substr(pos);
  size_t endHost = possibleHost.find_first_of(":/");
  std::string tmpHost = possibleHost.substr(0, endHost);
  if (CheckValidHost(tmpHost))
    host = tmpHost;
  else
    throw std::runtime_error("ParseURL(): invalid host '" + tmpHost + "'. "
        "Valid hostnames must contain only lowercase letters, digits,"
        " and hyphens.");

  if (endHost != std::string::npos)
  {
    char endChar = possibleHost.at(endHost);
    if (endChar == ':')
    {
      // We need to find the first '/' as we might have several.
      std::string possiblePort = possibleHost.substr(endHost + 1);
      size_t endPort = possiblePort.find_first_of("/");
      possiblePort = possiblePort.substr(0, endPort);
      if (IsDigits(possiblePort))
        port = std::stoi(possiblePort);
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

inline std::string GetCacheDir()
{
  std::filesystem::path dir;

#ifdef MLPACK_REMOTE_CACHE_DIR
  dir = MLPACK_REMOTE_CACHE_DIR;
#else
  #ifdef _WIN32
  char* appdata = std::getenv("APPDATA");
  if (appdata)
    dir = std::filesystem::path(appdata) / "mlpack" / "cache";
  #else
  char* home = std::getenv("HOME");
  if (home)
    dir = std::filesystem::path(home) / ".mlpack" / "cache";
  #endif

  if (dir.empty())
    dir = std::filesystem::temp_directory_path() / "mlpack_cache";
#endif

  std::error_code ec;
  std::filesystem::create_directories(dir, ec);

  if (ec)
  {
    Log::Fatal << "GetCacheDir(): cannot create cache directory '"
        << dir.string() << "': " << ec.message() << std::endl;
  }
  return dir.string();
}

inline bool LoadManifest(const std::string& cacheDir,
                         CacheManifest& manifest)
{
  std::string path =
      (std::filesystem::path(cacheDir) / "cache_manifest.csv").string();
  std::ifstream in(path);
  if (!in)
    return false;

  // Each line: url,etag,content_length,local_path
  std::string line;
  while (std::getline(in, line))
  {
    std::istringstream ss(line);
    std::string url, etag, sizeStr, localPath;
    if (std::getline(ss, url, ',') &&
        std::getline(ss, etag, ',') &&
        std::getline(ss, sizeStr, ',') &&
        std::getline(ss, localPath))
    {
      manifest[url] = std::make_tuple(etag, std::stoull(sizeStr), localPath);
    }
  }
  return true;
}

inline bool SaveManifest(const std::string& cacheDir,
                         const CacheManifest& manifest)
{
  std::string path =
      (std::filesystem::path(cacheDir) / "cache_manifest.csv").string();
  std::ofstream out(path, std::ios::trunc);

  if (!out)
  {
    Log::Warn << "SaveManifest(): cannot write cache manifest '"
        << path << "'." << std::endl;
    return false;
  }

  for (auto it = manifest.begin(); it != manifest.end(); ++it)
  {
    out << it->first << ","
        << std::get<0>(it->second) << ","
        << std::get<1>(it->second) << ","
        << std::get<2>(it->second) << "\n";
  }
  return true;
}

inline void WriteResponseToFile(const std::string& path,
                                const httplib::Result& res)
{
  std::fstream stream;
#ifdef _WIN32
  stream.open(path, std::fstream::out | std::fstream::binary);
#else
  stream.open(path, std::fstream::out);
#endif

  if (!stream.is_open())
  {
    Log::Fatal << "WriteResponseToFile(): cannot open file '" << path
        << "' for writing." << std::endl;
  }
  errno = 0;
  stream.write(res->body.data(), res->body.size());
  stream.flush();
  if (!stream.good())
  {
    Log::Fatal << "WriteResponseToFile(): error writing to '" << path
        << "': " << std::strerror(errno) << std::endl;
  }
  stream.close();
}

inline bool DownloadFile(const std::string& url, const std::string& dest)
{
  std::string host, filename;
  int port = -1;
  ParseURL(url, host, filename, port);

#ifdef MLPACK_USE_HTTPS
  if (port == -1)
    port = 443;
  httplib::SSLClient cli(host, port);
#else
  if (port == -1)
    port = 80;
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

  if (res->status != 200)
  {
    std::stringstream oss;
    oss << "DownloadFile(): HTTP request failed with status code "
        << res->status << " for URL '" << url << "'.";
    throw std::runtime_error(oss.str());
  }

  WriteResponseToFile(dest, res);
  return true;
}

inline bool DownloadFileWithCache(const std::string& url,
                                 std::string& filename)
{
  std::string host;
  int port = -1;
  ParseURL(url, host, filename, port);

#ifndef MLPACK_DISABLE_REMOTE_DATASET_CACHE
  std::string cacheDir = GetCacheDir();
  CacheManifest manifest;
  LoadManifest(cacheDir, manifest);

  std::string dest =
      (std::filesystem::path(cacheDir) / filename).string();

  // Do a head request to to get the cache info to compare it.
#ifdef MLPACK_USE_HTTPS
  if (port == -1)
    port = 443;
  httplib::SSLClient cli(host, port);
#else
  if (port == -1)
    port = 80;
  httplib::Client cli(host, port);
#endif
  cli.set_connection_timeout(2);

  // Register in cache only if the head request has succeeded.
  // If not just jump to download and do not cache the file, not enough
  // information to cache.
  std::string serverEtag;
  size_t serverSize = 0;
  bool headSucceeded = false;
  httplib::Result headRes = cli.Head(url);
  if (headRes && headRes->has_header("ETag"))
  {
    serverEtag = headRes->get_header_value("ETag");
    if (headRes->has_header("Content-Length"))
      serverSize = std::stoull(headRes->get_header_value("Content-Length"));
    headSucceeded = true;
  }

  if (headSucceeded &&
      manifest.count(url) > 0 &&
      std::get<0>(manifest[url]) == serverEtag &&
      std::get<1>(manifest[url]) == serverSize &&
      std::filesystem::exists(dest))
  {
    filename = dest;
    return true;
  }

  DownloadFile(url, dest);

  if (headSucceeded)
  {
    manifest[url] = std::make_tuple(serverEtag, serverSize, dest);
    SaveManifest(cacheDir, manifest);
  }
  filename = dest;
  return true;

#else

  // Download to a temporary file if cache is disabled.
  std::filesystem::path dest = TempName();
  dest += "." + Extension(filename);
  DownloadFile(url, dest.string());
  filename = dest.generic_string();
  return true;

#endif
}

#else

inline bool CheckIfURL(const std::string& /* url */)
{
  return false;
}

inline void ParseURL(const std::string& url, std::string& host,
                     std::string& filename, int& port)
{
  host = url;
  filename = "";
  port = -1;
  throw std::runtime_error("ParseURL(): httplib support not enabled; cannot "
      "check provided URL '" + url + "'.  Enable httplib by adding '#define "
      "MLPACK_ENABLE_HTTPLIB' before including mlpack.");
}

inline bool DownloadFile(const std::string& url,
                         const std::string& /* dest */)
{
  throw std::runtime_error("DownloadFile(): httplib support not enabled; cannot"
      " download URL '" + url + "'.  Enable httplib by adding '#define "
      "MLPACK_ENABLE_HTTPLIB' before including mlpack.");
  return false;
}

inline bool DownloadFileWithCache(const std::string& url,
                                 std::string& filename)
{
  filename = "";
  throw std::runtime_error("DownloadFile(): httplib support not enabled; cannot"
      " check provided URL '" + url + "'.  Enable httplib by adding '#define "
      "MLPACK_ENABLE_HTTPLIB' before including mlpack.");
  return false;
}

#endif

} // namespace mlpack

#endif

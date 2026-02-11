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

namespace mlpack {

inline void Decompress(const std::string& compressed, std::string decompressed)
{
  httplib::detail::gzip_decompressor decompressor;

  bool success = decompressor.decompress(compressed.data(),
      compressed.size(),
     [&decompressed](const char *data, size_t len)
     {
       decompressed.append(data, len);
       return true;
     });
}

template<typename DataOptionsType>
bool DownloadFile(const std::string& url,
                  std::string& filename,
                  std::fstream& stream,
                  DataOptionsType& opts)
{
  bool success = false;
  std::string originalFilename;
  FilenameFromURL(originalFilename, url);

#ifdef MLPACK_CACHE_REMOTE_DATASETS
  if (std::filesystem::exists(originalFilename))
  {
    std::filesystem::file_time_type fileTime =
        std::filesystem::last_write_time(originalFilename);

    std::chrono::time_point<std::filesystem::file_time_type::clock,
        FileTimeClock::duration> now = FileTimeClock::now();

    FileTimeClock::duration difference = now - fileTime;

    std::chrono::hours difference =
        std::chrono::duration_cast<std::chrono::hours>(difference);
    if (difference.count() > 5)
    {
      // only return, the file exist and we don't need to download it
      return true;
    }
  }
#endif

  // If host is not extracted correctly, we will get a segmentation fault from
  // httplib
  std::string host = URLToHost(url);
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  auto port = 443;
  httplib::SSLClient cli(host, port);
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
    return HandleError(oss, opts);
  }

  std::stringstream data(res->body);
  if (IsGzip(res->body.data()))
  {
// Check with rcurtin if we need to do the ifdef for gzip. probably not, as we
// would like to support to be by default.
// #ifndef MLPACK
    decompress(res->body.data(), data);
  }

#ifdef MLPACK_CACHE_REMOTE_DATASETS
  success = WriteToFile(originalFilename, opts, data.str(), stream);
#else
  // This does not work, please see above.
  // filename = std::tmpnam(nullptr);

  filename = TempName();
  // This is necessary to get the extension.
  filename += originalFilename;
  success = OpenFile(filename, opts, false, stream);
  if (!success)
  {
    std::stringstream oss;
    oss <<  "Unable to open a temporary file for downloading data.";
    return HandleError(oss, opts);
  }

  stream.write(data.str().data(), data.str().size());
  if (!stream.good())
  {
    std::stringstream oss;
    oss << "Error writing to a '" << filename << "'.  "
          << "Please check permissions or disk space.";
    return HandleError(oss, opts);
  }
  stream.close();
#endif
  return success;
}

} // namespace mlpack

#endif

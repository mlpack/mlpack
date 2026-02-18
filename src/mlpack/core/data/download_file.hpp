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

//@rcurtin, it seems that using tmpnam is a bad idea, since the software was
//sefgaulting and the compiler was throwing the following warning:
//
// (.text+0x83d4b): warning: the use of `tmpnam' is dangerous, better use `mkstemp'  
//
//
// After checking online, a lot of people and discussion pointed out to avoid
// using std::tmpname since it is not safe in the case of race condition.
//
//
// https://stackoverflow.com/questions/78535907/can-we-implement-a-facility-to-safely-create-temporary-files-in-c23
//
// https://stackoverflow.com/questions/75867045/temporary-files-in-c-tmpnam-alternatives
//
// https://stackoverflow.com/questions/35188145/warning-the-use-of-tmpnam-is-dangerous-better-use-mkstemp
//
// I have also seen some stupid proposals, saying, just ignore or disable the
// compiler warning. 
//
// The main problem with the compiler proposal of using mkstemp is the need
// to define the name of the temporary file, which defies the first reason we used this one.
//
// Here is a code sinppet that might looks like to use mkstemp():
// char filename[] = "/tmp/tempXXXXXX";
// int fd = mkstemp(filename);
// if (fd != -1)
// {
//   std::cout << "random file opened" << std::endl;
// }
// 
// Assuming that the directory problem is solved on Windows, the above solution 
// would segfault sometimes and run normally other times, it depends on the
// runs. This was not really constructive to debug with gdb, so I abadndone
// this idea, as I thought, there should be probably a better way that is used
// by everyone to solve this simple problem, and here is the solution in the
// following function:
//
// For the following function , I have adapted the following implementation
// from C++20 to C++17 below:
//
// https://codereview.stackexchange.com/questions/292241/generate-unique-temporary-file-names-in-c20
//
// If this works on Windows, I would recommend using it as it is, as
// it is going to be much easier to be handle with our current infrastrucutre
// without the need to use any C function. 
//
std::filesystem::path TempName()
{
  static std::mt19937 gen{std::random_device{}()};
  static std::uniform_int_distribution<>
      dist{0, std::numeric_limits<uint8_t>::max()};
  std::stringstream nameStream;
  // i.e. long enough to avoid collisions (see UUID)
  static constexpr auto num_bits = 128;
  for (size_t i = 0; i < (num_bits / std::numeric_limits<uint8_t>::digits); ++i)
  {
      nameStream << dist(gen);
  }
  return std::filesystem::temp_directory_path() / nameStream.str();
}

template<typename DataOptionsType>
bool DownloadFile(const std::string& url,
                  std::string& filename,
                  std::fstream& stream,
                  DataOptionsType& opts)
{
  bool success = false;
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
    return HandleError(oss, opts);
  }

  std::stringstream data(res->body);
  std::string originalFilename;
  FilenameFromURL(originalFilename, url);

//#ifdef MLPACK_CAHCHE
  //success = WriteToFile(filename, opts, data.str(), stream);
//#endif
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
  return success;
}

} // namespace mlpack

#endif

#endif

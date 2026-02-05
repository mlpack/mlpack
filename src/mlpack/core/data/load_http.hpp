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


// includes

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

template<typename MatType>
bool LoadHTTP(const std::string& url,
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
    return HandleError(oss, opts)
  }

  //std::cout << "Show the body of the message:"  << std::endl << res->body
   //   << std::endl;
  std::stringstream data(res->body);

  FilenameFromURL(filename, url);
  if (opts.EnableCache())
  {
    success = WriteToFile(filename, opts, data.str(), stream);
  }
  else
  {
    // problems with tmpfile()
    // It seems that using tmpfile() is a not good for the following reasons reasons.
    // 1. First the file created with automatically generated name, also the
    // file does not appear in the filesystem and it is fully held by the
    // current process and destroyed when the process is destroyed.
    // 2. tmpnam() exists so we can generate the temporary name, but we cannot
    // use this temporary name with tmpfile()
    // 3. tmpfile() return a file descriptor and there is no way to change the
    // file descriptor to std::fstream. This will case a problem with the
    // current functions.
    //std::FILE* tmp = std::tmpfile();
    //
    // A solution is the following (Not possible, but it was explored):
    //
    // 1. Use the same filename that we extract from the downloaded file
    // 2. Get the temprorary location of the system where the tempfiles are
    // located using fs::temp_directory_path()
    // 3. Create the temporary file using int fd = std::mkstemp(filename);
    // 4. Write the data to the file using write, or fput (better write with fd)
    //  It seems that it is not possible to use write because this is OS
    //  relevant function
    //  Also it seems that mkstemp() cannot be used with fputs because it
    //  returns an int.
    // 5. Once this is done close(fd) the file and only keep the file name for the
    // Load function. Note, at this stage the file appears in the filesystem
    // and it is not destroyed.
    // 6. Open the file and load it using Load() interface.
    // 7. Optional, we can delete this file using std::unlink(filename); Once
    // we finished the manipulation process, but this is not obligatory.
    //


    // Another solution that is possible
    // We follow the same above logic, until we arrive to temporary file
    // creation.
    //
    // In this case, we open and create the file in the temporary directory, in
    // the same way we do at mlpack using std::open. fstream, etc.
    // We write to that file, using the write function that I have created.
    // Then we return the std::fstream so we can have access to the file to
    // load it if we are using CSV, or just return the filename so we can load
    // it with STB in the case of an image.
    //
    // Finally we call ulink on the file if we are not interested in keeping it

  }


  return success;
}

} // namespace mlpack

#endif

#endif

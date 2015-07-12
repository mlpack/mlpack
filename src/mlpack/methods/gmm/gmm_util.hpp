/**
 * @file gmm_util.hpp
 * @author Ryan Curtin
 *
 * Utility to save GMMs to files.
 */
#ifndef __MLPACK_METHODS_GMM_GMM_UTIL_HPP
#define __MLPACK_METHODS_GMM_GMM_UTIL_HPP

namespace mlpack {
namespace gmm {

// Save a GMM to file using boost::serialization.
// This does not save a type id, however.
template<typename GMMType>
void SaveGMM(GMMType& g, const std::string filename)
{
  using namespace boost::archive;

  const std::string extension = data::Extension(filename);
  std::ofstream ofs(filename);
  if (extension == "xml")
  {
    xml_oarchive ar(ofs);
    ar << data::CreateNVP(g, "gmm");
  }
  else if (extension == "bin")
  {
    binary_oarchive ar(ofs);
    ar << data::CreateNVP(g, "gmm");
  }
  else if (extension == "txt")
  {
    text_oarchive ar(ofs);
    ar << data::CreateNVP(g, "gmm");
  }
  else
    Log::Fatal << "Unknown extension '" << extension << "' for GMM model file "
        << "(known: 'xml', 'bin', 'txt')." << std::endl;
}

} // namespace gmm
} // namespace mlpack

#endif

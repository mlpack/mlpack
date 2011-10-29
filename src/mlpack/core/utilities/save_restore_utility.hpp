/**
 * @file utilities/save_restore_utility.hpp
 * @author Neil Slagle
 *
 * The SaveRestoreUtility provides helper functions in saving and
 *   restoring models.  The current output file type is XML.
 *
 * @experimental
 */

#ifndef SAVE_RESTORE_MODEL_HPP
#define SAVE_RESTORE_MODEL_HPP

#include <err.h>
#include <list>
#include <map>
#include <sstream>
#include <string>

#include <libxml/parser.h>
#include <libxml/tree.h>

#include <mlpack/core.h>
#include <boost/tokenizer.hpp>

namespace mlpack {
namespace utilities {

class SaveRestoreUtility
{
 private:
  /**
   * parameters contains a list of names and parameters in string form.
   */
  std::map<std::string, std::string> parameters;
  /**
   * RecurseOnNodes performs a depth first search of the XML tree.
   */
  void RecurseOnNodes (xmlNode* n);
 public:
  SaveRestoreUtility() {}
  ~SaveRestoreUtility() { parameters.clear(); }
  /**
   * ReadFile reads an XML tree from a file.
   */
  bool ReadFile (std::string filename);
  /**
   * WriteFile writes the XML tree to a file.
   */
  bool WriteFile (std::string filename);
  /**
   * LoadParameter loads a parameter from the parameters map.
   */
  template<typename T>
  T& LoadParameter (T& t, std::string name);
  /**
   * LoadParameter loads a character from the parameters map.
   */
  char LoadParameter (char c, std::string name);
  /**
   * LoadParameter loads a string from the parameters map.
   */
  std::string LoadParameter (std::string str, std::string name);
  /**
   * LoadParameter loads an arma::mat from the parameters map.
   */
  arma::mat& LoadParameter (arma::mat& matrix, std::string name);
  /**
   * SaveParameter saves a parameter to the parameters map.
   */
  template<typename T>
  void SaveParameter (T& t, std::string name);
  /**
   * SaveParameter saves a character to the parameters map.
   */
  void SaveParameter (char c, std::string name);
  /**
   * SaveParameter saves an arma::mat to the parameters map.
   */
  void SaveParameter (arma::mat& mat, std::string name);
};
} /* namespace model */
} /* namespace mlpack */

#include "save_restore_utility_impl.hpp"

#endif

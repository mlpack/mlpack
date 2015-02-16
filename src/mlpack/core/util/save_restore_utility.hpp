/**
 * @file save_restore_utility.hpp
 * @author Neil Slagle
 *
 * The SaveRestoreUtility provides helper functions in saving and
 *   restoring models.  The current output file type is XML.
 *
 * @experimental
 */
#ifndef __MLPACK_CORE_UTIL_SAVE_RESTORE_UTILITY_HPP
#define __MLPACK_CORE_UTIL_SAVE_RESTORE_UTILITY_HPP

#include <mlpack/prereqs.hpp>
#include <list>
#include <map>
#include <sstream>
#include <string>

#include <libxml/parser.h>
#include <libxml/tree.h>

#include <boost/tokenizer.hpp>

namespace mlpack {
namespace util {

class SaveRestoreUtility
{
 private:
  /**
   * parameters contains a list of names and parameters in string form.
   */
  std::map<std::string, std::string> parameters;

  /**
   * children contains a list of names in string format and child
   * models in the model hierarchy in SaveRestoreUtility format
   */
  std::map<std::string, SaveRestoreUtility> children;

  /**
   * RecurseOnNodes performs a depth first search of the XML tree.
   */
  void RecurseOnNodes(xmlNode* n);

 public:

  SaveRestoreUtility() {}
  ~SaveRestoreUtility() { parameters.clear(); }

  /**
   * ReadFile reads an XML tree from a file.
   */
  bool ReadFile(const std::string& filename);

  /**
   * WriteFile writes the XML tree to a file.
   */
  bool WriteFile(const std::string& filename);

  /**
   * LoadParameter loads a parameter from the parameters map.
   */
  template<typename T>
  T& LoadParameter(T& t, const std::string& name) const;

  /**
   * LoadParameter loads a parameter from the parameters map.
   */
  template<typename T>
  std::vector<T>& LoadParameter(std::vector<T>& v, const std::string& name)
      const;

  /**
   * LoadParameter loads a character from the parameters map.
   */
  char LoadParameter(char c, const std::string& name) const;

  /**
   * LoadParameter loads a string from the parameters map.
   */
  std::string LoadParameter(std::string& str, const std::string& name) const;

  /**
   * LoadParameter loads an arma::mat from the parameters map.
   */
  arma::mat& LoadParameter(arma::mat& matrix, const std::string& name) const;

  /**
   * SaveParameter saves a parameter to the parameters map.
   */
  template<typename T>
  void SaveParameter(const T& t, const std::string& name);

  /**
   * SaveParameter saves a parameter to the parameters map.
   */
  template<typename T>
  void SaveParameter(const std::vector<T>& v, const std::string& name);

  /**
   * SaveParameter saves a character to the parameters map.
   */
  void SaveParameter(const char c, const std::string& name);

  /**
   * SaveParameter saves an arma::mat to the parameters map.
   */
  void SaveParameter(const arma::mat& mat, const std::string& name);

  /**
   * SaveSubModel saves a SaveRestoreUtility to the children map.
   */
  void AddChild(SaveRestoreUtility& mn, const std::string& name);

  /**
   * Return the children.
   */
  const std::map<std::string, SaveRestoreUtility> Children() const { return
    children; }

  /**
   * Return modifiable copy of the children.
   */
  std::map<std::string, SaveRestoreUtility> Children() { return children; }

 private:
  /**
   * WriteFile creates XML tree recursively.
   */
  void WriteFile(xmlNode* n);

  /**
   * ReadFile reads an XML tree recursively.
   */
  void ReadFile(xmlNode* n);
};

//! Specialization for arma::vec.
template<>
arma::vec& SaveRestoreUtility::LoadParameter(arma::vec& t,
                                             const std::string& name) const;

//! Specialization for arma::vec.
template<>
void SaveRestoreUtility::SaveParameter(const arma::vec& t,
                                       const std::string& name);



}; /* namespace util */
}; /* namespace mlpack */

// Include implementation.
#include "save_restore_utility_impl.hpp"

#endif

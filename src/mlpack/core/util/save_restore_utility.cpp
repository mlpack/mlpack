/**
 * @file save_restore_utility.cpp
 * @author Neil Slagle
 *
 * The SaveRestoreUtility provides helper functions in saving and
 *   restoring models.  The current output file type is XML.
 *
 * This file is part of MLPACK 1.0.11.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <mlpack/core.hpp>

using namespace mlpack;
using namespace mlpack::util;

bool SaveRestoreUtility::ReadFile(const std::string& filename)
{
  xmlDocPtr xmlDocTree = NULL;
  if (NULL == (xmlDocTree = xmlReadFile(filename.c_str(), NULL, 0)))
  {
    Log::Fatal << "Could not load XML file '" << filename << "'!" << std::endl;
  }

  xmlNodePtr root = xmlDocGetRootElement(xmlDocTree);
  parameters.clear();

  RecurseOnNodes(root->children);
  xmlFreeDoc(xmlDocTree);
  return true;
}

void SaveRestoreUtility::RecurseOnNodes(xmlNode* n)
{
  xmlNodePtr current = NULL;
  for (current = n; current; current = current->next)
  {
    if (current->type == XML_ELEMENT_NODE)
    {
      xmlChar* content = xmlNodeGetContent(current);
      parameters[(const char*) current->name] = (const char*) content;
      xmlFree(content);
    }
    RecurseOnNodes(current->children);
  }
}

bool SaveRestoreUtility::WriteFile(const std::string& filename)
{
  bool success = false;
  xmlDocPtr xmlDocTree = xmlNewDoc(BAD_CAST "1.0");
  xmlNodePtr root = xmlNewNode(NULL, BAD_CAST "root");

  xmlDocSetRootElement(xmlDocTree, root);

  for (std::map<std::string, std::string>::iterator it = parameters.begin();
       it != parameters.end();
       ++it)
  {
    xmlNewChild(root, NULL, BAD_CAST(*it).first.c_str(),
                            BAD_CAST(*it).second.c_str());
    /* TODO: perhaps we'll add more later?
     * xmlNewProp(child, BAD_CAST "attr", BAD_CAST "add more addibutes?"); */
  }

  // Actually save the file.
  success =
      (xmlSaveFormatFileEnc(filename.c_str(), xmlDocTree, "UTF-8", 1) != -1);
  xmlFreeDoc(xmlDocTree);
  return success;
}

arma::mat& SaveRestoreUtility::LoadParameter(arma::mat& matrix,
                                             const std::string& name)
{
  std::map<std::string, std::string>::iterator it = parameters.find(name);
  if (it != parameters.end())
  {
    std::string value = (*it).second;
    boost::char_separator<char> sep ("\n");
    boost::tokenizer<boost::char_separator<char> > tok (value, sep);
    std::list<std::list<double> > rows;
    for (boost::tokenizer<boost::char_separator<char> >::iterator
           tokIt = tok.begin();
         tokIt != tok.end();
         ++tokIt)
    {
      std::string row = *tokIt;
      boost::char_separator<char> sepComma (",");
      boost::tokenizer<boost::char_separator<char> >
        tokInner (row, sepComma);
      std::list<double> rowList;
      for (boost::tokenizer<boost::char_separator<char> >::iterator
             tokInnerIt = tokInner.begin();
             tokInnerIt != tokInner.end();
             ++tokInnerIt)
      {
        double element;
        std::istringstream iss (*tokInnerIt);
        iss >> element;
        rowList.push_back(element);
      }
      rows.push_back(rowList);
    }
    matrix.zeros(rows.size(), (*(rows.begin())).size());
    size_t rowCounter = 0;
    size_t columnCounter = 0;
    for (std::list<std::list<double> >::iterator rowIt = rows.begin();
         rowIt != rows.end();
         ++rowIt)
    {
      std::list<double> row = *rowIt;
      columnCounter = 0;
      for (std::list<double>::iterator elementIt = row.begin();
           elementIt != row.end();
           ++elementIt)
      {
        matrix(rowCounter, columnCounter) = *elementIt;
        columnCounter++;
      }
      rowCounter++;
    }
    return matrix;
  }
  else
  {
    Log::Fatal << "LoadParameter(): node '" << name << "' not found.\n";
  }
  return matrix;
}

std::string SaveRestoreUtility::LoadParameter(std::string& str,
                                              const std::string& name)
{
  std::map<std::string, std::string>::iterator it = parameters.find(name);
  if (it != parameters.end())
  {
    return str = (*it).second;
  }
  else
  {
    Log::Fatal << "LoadParameter(): node '" << name << "' not found.\n";
  }
  return "";
}

char SaveRestoreUtility::LoadParameter(char c, const std::string& name)
{
  std::map<std::string, std::string>::iterator it = parameters.find(name);
  if (it != parameters.end())
  {
    int temp;
    std::string value = (*it).second;
    std::istringstream input (value);
    input >> temp;
    return c = (char) temp;
  }
  else
  {
    Log::Fatal << "LoadParameter(): node '" << name << "' not found.\n";
  }
  return 0;
}

void SaveRestoreUtility::SaveParameter(const char c, const std::string& name)
{
  int temp = (int) c;
  std::ostringstream output;
  output << temp;
  parameters[name] = output.str();
}

void SaveRestoreUtility::SaveParameter(const arma::mat& mat,
                                       const std::string& name)
{
  std::ostringstream output;
  size_t columns = mat.n_cols;
  size_t rows = mat.n_rows;
  for (size_t r = 0; r < rows; ++r)
  {
    for (size_t c = 0; c < columns - 1; ++c)
    {
      output << std::setprecision(15) << mat(r, c) << ",";
    }
    output << std::setprecision(15) << mat(r, columns - 1) << std::endl;
  }
  parameters[name] = output.str();
}

// Special template specializations for vectors.

namespace mlpack {
namespace util {

template<>
arma::vec& SaveRestoreUtility::LoadParameter(arma::vec& t,
                                             const std::string& name)
{
  return (arma::vec&) LoadParameter((arma::mat&) t, name);
}

template<>
void SaveRestoreUtility::SaveParameter(const arma::vec& t,
                                       const std::string& name)
{
  SaveParameter((const arma::mat&) t, name);
}

}; // namespace util
}; // namespace mlpack

/**
 * @file save_restore_utility.cpp
 * @author Neil Slagle
 * @author Michael Fox
 *
 * The SaveRestoreUtility provides helper functions in saving and
 *   restoring models.  The current output file type is XML.
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
  ReadFile(root->children);
  xmlFreeDoc(xmlDocTree);
  return true;
}

void SaveRestoreUtility::ReadFile(xmlNode* n)
{
  parameters.clear();
  xmlNodePtr current = NULL;
  for (current = n; current; current = current->next)
  {
    if (current->type == XML_ELEMENT_NODE)
    {
      xmlChar* content = xmlNodeGetContent(current);
      if(xmlChildElementCount(current) == 0)
      {
        parameters[(const char*) current->name] = (const char*) content;
      }
      else
      {
        children[(const char*) current->name].ReadFile(current->children);
      }
      xmlFree(content);
    }
  }
}

bool SaveRestoreUtility::WriteFile(const std::string& filename)
{
  bool success = false;
  xmlDocPtr xmlDocTree = xmlNewDoc(BAD_CAST "1.0");
  xmlNodePtr root = xmlNewNode(NULL, BAD_CAST "root");
  xmlDocSetRootElement(xmlDocTree, root);
  WriteFile(root);

  // Actually save the file.
  success = (xmlSaveFormatFileEnc(filename.c_str(), xmlDocTree, "UTF-8", 1) !=
             -1);
  xmlFreeDoc(xmlDocTree);
  return success;
}

void SaveRestoreUtility::WriteFile(xmlNode* n)
{
  for (std::map<std::string, std::string>::reverse_iterator it =
	    parameters.rbegin(); it != parameters.rend(); ++it)
  {
    xmlNewChild(n, NULL, BAD_CAST(*it).first.c_str(),
        BAD_CAST(*it).second.c_str());
  }
  xmlNodePtr child;
  for (std::map<std::string, SaveRestoreUtility>::iterator it =
       children.begin(); it != children.end(); ++it)
  {
    child = xmlNewChild(n, NULL, BAD_CAST(*it).first.c_str(), NULL);
    it->second.WriteFile(child);
  }
}

arma::mat& SaveRestoreUtility::LoadParameter(arma::mat& matrix,
                                             const std::string& name) const
{
  std::map<std::string, std::string>::const_iterator it = parameters.find(name);
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
                                              const std::string& name) const
{
  std::map<std::string, std::string>::const_iterator it = parameters.find(name);
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

char SaveRestoreUtility::LoadParameter(char c, const std::string& name) const
{
  std::map<std::string, std::string>::const_iterator it = parameters.find(name);
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
                                             const std::string& name) const
{
  return (arma::vec&) LoadParameter((arma::mat&) t, name);
}

template<>
void SaveRestoreUtility::SaveParameter(const arma::vec& t,
                                       const std::string& name)
{
  SaveParameter((const arma::mat&) t, name);
}
    
void SaveRestoreUtility::AddChild(SaveRestoreUtility& mn, const std::string&
    name)
{
  children[name] = mn;
}

  
}; // namespace util
}; // namespace mlpack

/**
 * @file optionshierarchy.cpp
 * @author Matthew Amidon
 *
 * Implementation of OptionsHierarchy class.
 */
#include <iomanip>
#include <iostream>

#include "cli.hpp"
#include "log.hpp"
#include "optionshierarchy.hpp"

using namespace std;
using namespace mlpack::io;

/* Ctors, Dtors, and R2D2 [actually, just copy-tors] */

/* Constructs an empty OptionsHierarchy node. */
OptionsHierarchy::OptionsHierarchy() : children()
{
  nodeData.node = "";
  nodeData.desc = "";
  nodeData.tname = "";
  return;
}

/*
 * Constructs an empty OptionsHierarchy node
 *
 * @param name The name of the node to be created.
 */
OptionsHierarchy::OptionsHierarchy(const char* name) : children()
{
  nodeData.node = string(name);
  nodeData.desc = "";
  nodeData.tname = "";
  return;
}

/*
 * Constructs an equivalent node to the given one.
 *
 * @param other The node to be copied
 */
OptionsHierarchy::OptionsHierarchy(const OptionsHierarchy& other)
{
  return;
}

/*
 * Destroys the node.
 */
OptionsHierarchy::~OptionsHierarchy()
{
  return;
}

/*
 * Will never fail, as given paths are relative to current node
 * and will be generated if not found.
 *
 * @param pathname The full pathname of the given node, eg /foo/bar.
 * @param tname A string unique to the type of the node.
 */
void OptionsHierarchy::AppendNode(string& pathname, string& tname)
{
  string tmp = string("");
  OptionsData d;
  d.node = pathname;
  d.desc = tmp;
  d.tname = tname;
  AppendNode(pathname, tname, tmp, d);
}

/*
 * Will never fail, as given paths are relative to current node
 * and will be generated if not found.
 *
 * @param pathname The full pathname of the given node, eg /foo/bar.
 * @param tname A string unique to the type of the node.
 * @param description String description of the node.
 */
void OptionsHierarchy::AppendNode(string& pathname,
                                  string& tname,
                                  string& description)
{
  OptionsData d;
  d.node = pathname;
  d.desc = description;
  d.tname = tname;
  AppendNode(pathname, tname, description, d);
}

/*
 * Will never fail, as given paths are relative to current node
 * and will be generated if not found.
 *
 * @param pathname The full pathname of the given node, eg /foo/bar.
 * @param tname A string unique to the type of the node.
 * @param description String description of the node.
 * @param data Specifies all fields of the new node.
 */
void OptionsHierarchy::AppendNode(string& pathname, string& tname,
                                  string& description, OptionsData& data)
{
  string name = GetName(pathname);
  string path = GetPath(pathname);
  //Append the new name, if it isn't already there
  if (children.count(name) == 0)
    children[name] = OptionsHierarchy(name.c_str());

  if (pathname.find('/') == pathname.npos || path.length() < 1)
  {
    children[name].nodeData = data;
    return;
  }

  //Recurse until path is done
  children[name].AppendNode(path, tname, description, data);
}
/*
 * Will return the node associated with a pathname
 *
 * @param pathname The full pathname of the node,
 *   eg foo/bar in foo/bar.
 *
 * @return Pointer to the node with that pathname,
 *   null if not found.
 */
OptionsHierarchy* OptionsHierarchy::FindNode(string& pathname)
{
  return FindNodeHelper(pathname, pathname);
}

OptionsHierarchy* OptionsHierarchy::FindNodeHelper(string& pathname,
                                                  string& target)
{
  string name = GetName(pathname);
  string path = GetPath(pathname);
  //If the node is there, recurse to it.
  if (path.length() != 0 || name.length() != 0)
    return children[name].FindNodeHelper(path, target);

  if (target.compare(nodeData.node) == 0)
   return this;

  return NULL;
}

/*
 * Returns the various data associated with a node.  Passed by copy,
 * since this is only for unit testing.
 *
 * @return The data associated with the node,
 * eg it's name, description, and value.
 */
OptionsData OptionsHierarchy::GetNodeData()
{
  return nodeData;
}

/* Returns the path bar/fizz in the pathname foo/bar/fizz
  *
  * @param pathname The full pathname of the parameter,
  *   eg foo/bar in foo/bar.
  *
  * @return The identifiers of all nodes after the next node in the path,
  *   eg fizz/bar in foo/fizz/bar.
  */
string OptionsHierarchy::GetPath(string& pathname)
{
  //Want to make sure we return a valid string
  if (pathname.find('/') == pathname.npos)
    return string("");
  //Get the rest of the node name Eg foo/bar in root/foo/bar
  return pathname.substr(pathname.find('/')+1,pathname.length());
}

/* Returns the name foo in the pathname foo/bar/fizz
 *
 * @param pathname The full pathname of the parameter,
 *   eg foo/bar in foo/bar.
 *
 * @return The name of the next node in the path
 *   eg foo in foo/bar.
 */
string OptionsHierarchy::GetName(string& pathname)
{
  //Want to makesure we return a valid string
  if (pathname.find('/') == pathname.npos)
    return pathname;
  //Get the topmost node name in this path Eg root in root/foo/bar
  return pathname.substr(0, pathname.find('/'));
}


/*
 * Obtains a vector containing relative pathnames of nodes subordinant to
 * the one specified in the parameter.
 *
 * @param pathname The full pathname to the node in question.
 *
 * @return Vector containing relative pathnames of subordinant nodes.
 */
std::vector<std::string>
  OptionsHierarchy::GetRelativePaths(std::string& pathname)
{
  std::vector<std::string> ret;

  //Obtain the starting node.
  OptionsHierarchy* node = FindNode(pathname);
  if(node == NULL)
    return ret;

  //Start adding it's children etc.
  return GetRelativePathsHelper(*node);
}

std::vector<std::string>
  OptionsHierarchy::GetRelativePathsHelper(OptionsHierarchy& node)
{
  std::vector<std::string> ret;
  std::vector<std::string> tmp;

  tmp.push_back(node.nodeData.node);
  ChildMap::iterator iter;
  for(iter = node.children.begin(); iter != node.children.end(); iter++)
    tmp = GetRelativePathsHelper((*iter).second);

  while(tmp.size())
  {
    ret.push_back(tmp.back());
    tmp.pop_back();
  }

  return ret;
}
/*
 * Prints a node, followed by it's entries and submodules.
 */
void OptionsHierarchy::Print()
{
  //Print the node, append '/' if that node is not a leaf
  PrintNode();

  //Begin formatted output
  cout << "Entries:" << endl;
  PrintLeaves();

  cout << "Submodules:" << endl;
  PrintBranches();
}

/*
 * Prints every node and it's value, if any.
 */
void OptionsHierarchy::PrintAll()
{
  PrintNode();
  map<string, OptionsHierarchy>::iterator iter;
  for (iter = children.begin(); iter != children.end(); iter++)
  {
    iter->second.PrintAll();
  }
}

/*
 * Prints every node and it's description.
 */
void OptionsHierarchy::PrintAllHelp()
{
  // Special case for the top of the hierarchy.
  if (nodeData.node == "Allowed Options")
    cout << "Allowed Options:" << endl << endl;
  else
    PrintNodeHelp();

  // Now print all the children.
  map<string, OptionsHierarchy>::iterator iter;
  // First print modules.
  for (iter = children.begin(); iter != children.end(); iter++)
  {
    if (iter->second.children.size() > 0)
      iter->second.PrintAllHelp();
  }

  // Now print leaves.
  // If this is the root node, we have to mention that these are default
  // options.
  if (nodeData.node == "Allowed Options")
    cout << "Other options:" << endl << endl;

  for (iter = children.begin(); iter != children.end(); iter++)
  {
    if (iter->second.children.size() == 0)
      iter->second.PrintAllHelp();
  }

  if (children.size() > 0) // If this was a module.
    cout << endl; // Newline for separation from other modules.
}

/* Prints all children of this node which are parents */
void OptionsHierarchy::PrintBranches()
{
  map<string, OptionsHierarchy>::iterator iter;

  // Iterate through all children
  for (iter = children.begin(); iter != children.end(); iter++)
  // Does this child have children?
    if (iter->second.children.size())
    {
      iter->second.PrintNode();
    }
}


/* Prints all children nodes that have no children themselves */
void OptionsHierarchy::PrintLeaves()
{
  map<string, OptionsHierarchy>::iterator iter;
  for (iter = children.begin(); iter != children.end(); iter++)
  {
    if (!iter->second.children.size())
    {
      // Print the node's name, data, and description, but only if it is not a
      // timer.
      if (iter->second.nodeData.tname != TYPENAME(timeval))
        iter->second.PrintNode();
    } else
    {
      iter->second.PrintLeaves();
    }
  }
}

/* Prints all children nodes that are timers */
void OptionsHierarchy::PrintTimers()
{
  map<string, OptionsHierarchy>::iterator iter;
  for (iter = children.begin(); iter != children.end(); iter++)
  {
    if (!iter->second.children.size())
    {
      if (iter->second.nodeData.tname == TYPENAME(timeval))
        iter->second.PrintNode();
    } else
    {
      iter->second.PrintTimers();
    }
  }
}

/*
 * Prints a node and its value.
 */
void OptionsHierarchy::PrintNode()
{
  Log::Info << "  " << nodeData.node << " = " ;

  if (nodeData.tname == TYPENAME(bool))
    Log::Info << boolalpha << CLI::GetParam<bool>(nodeData.node.c_str());
  else if (nodeData.tname == TYPENAME(int))
    Log::Info << CLI::GetParam<int>(nodeData.node.c_str());
  else if (nodeData.tname == TYPENAME(std::string))
  {
    std::string value = CLI::GetParam<std::string>(nodeData.node.c_str());
    if (value == "")
      value = "\"\""; // So that the user isn't presented with an empty space.
    Log::Info << value;
  } else if (nodeData.tname == TYPENAME(float))
    Log::Info << CLI::GetParam<float>(nodeData.node.c_str());
  else if (nodeData.tname == TYPENAME(double))
    Log::Info << CLI::GetParam<double>(nodeData.node.c_str());
  else if (nodeData.tname == TYPENAME(timeval))
  {
    timeval& t = CLI::GetParam<timeval>(nodeData.node.c_str());
    Log::Info << t.tv_sec << "." << std::setw(6) << std::setfill('0')
        << t.tv_usec << "s";
    // Also output convenient day/hr/min/sec.
    int days = t.tv_sec / 86400; // Integer division rounds down.
    int hours = (t.tv_sec % 86400) / 3600;
    int minutes = (t.tv_sec % 3600) / 60;
    int seconds = (t.tv_sec % 60);

    // No output if it didn't even take a minute.
    if (!(days == 0 && hours == 0 && minutes == 0))
    {
      bool output = false; // Denotes if we have output anything yet.
      Log::Info << " (";
      // Only output units if they have nonzero values (yes, a bit tedious).
      if (days > 0)
      {
        Log::Info << days << " days";
        output = true;
      }
      if (hours > 0)
      {
        if (output)
          Log::Info << ", ";
        Log::Info << hours << " hrs";
        output = true;
      }
      if (minutes > 0)
      {
        if (output)
          Log::Info << ", ";
        Log::Info << minutes << " mins";
        output = true;
      }
      if (seconds > 0)
      {
        if (output)
          Log::Info << ", ";
        Log::Info << seconds << "." << std::setw(1) << (t.tv_usec / 100000) <<
            " secs";
        output = true;
      }

      Log::Info << ")";
    }
  }

  Log::Info << endl;
}

/*
 * Prints a node and its description.  The format is similar to that help given
 * by the ImageMagick suite of programs.
 */
void OptionsHierarchy::PrintNodeHelp()
{
  // We want to print differently if this is a module node (i.e. if it has any
  // children).
  if (children.size() > 0)
  {
    if (nodeData.node == "default") // Special case for default module.
      cout << "Default options:" << endl;
    else // Other standard module title output.
      cout << '\'' << nodeData.node << "' module: " << endl;
    cout << "  ";
    if (nodeData.desc.length() > 0)
      cout << HyphenateString(nodeData.desc, 2) << endl << endl;
    else
      cout << "Undocumented module." << endl << endl;

    return; // Nothing else to do.
  }

  // Name of node gets printed first, with two spaces in front.
  // If there is a parameter, we specify that below.  We keep track of the
  // length of what we've written.
  cout << "  --" << nodeData.node << " ";
  int len = 5 + nodeData.node.length();

  // Perhaps this should be moved somewhere more central, as it may need to be
  // used more than just here.
  string value = "[value]";
  if (nodeData.tname == TYPENAME(bool))
    value = "";
  else if (nodeData.tname == TYPENAME(int))
    value = "[int]";
  else if (nodeData.tname == TYPENAME(float))
    value = "[float]";
  else if (nodeData.tname == TYPENAME(double))
    value = "[double]";
  else if (nodeData.tname == TYPENAME(std::string))
    value = "[string]";

  cout << value;
  len += value.length();

  // So, we only want to use a new line if we have used more than 30 characters
  // already.  Descriptions start at character 30.
  if (len < 30)
  {
    cout << std::string(30 - len, ' ');
    if (nodeData.desc.length() > 0)
      cout << HyphenateString(nodeData.desc, 30) << endl;
    else
      cout << "Undocumented option." << endl;
  } else
  {
    cout << endl << std::string(30, ' ');
    if (nodeData.desc.length() > 0)
      cout << HyphenateString(nodeData.desc, 30) << endl;
    else
      cout << "Undocumented option." << endl;
  }
}

/**
 * Hyphenate a string or split it onto multiple 80-character lines, with some
 * amount of padding on each line.  This is used for option output.
 *
 * @param str String to hyphenate (splits are on ' ').
 * @param padding Amount of padding on the left for each new line.
 */
string OptionsHierarchy::HyphenateString(string str, int padding)
{
  size_t margin = 80 - padding;
  if (str.length() < margin)
    return str;

  string out("");
  unsigned int pos = 0;

  // First try to look as far as possible.
  while(pos < str.length() - 1)
  {
    size_t splitpos;
    // Check that we don't have a newline first.
    splitpos = str.find('\n', pos);
    if (splitpos == string::npos || splitpos > (pos + margin))
    {
      // We did not find a newline.
      if (str.length() - pos < margin)
      {
        splitpos = str.length(); // The rest fits on one line.
      } else
      {
        splitpos = str.rfind(' ', margin + pos); // Find nearest space.
        if (splitpos <= pos || splitpos == string::npos) // Not found.
          splitpos = pos + margin;
      }
    }

    out += str.substr(pos, (splitpos - pos));
    if (splitpos < str.length())
    {
      out += '\n';
      out += string(padding, ' ');
    }

    pos = splitpos;
    if (str[pos] == ' ' || str[pos] == '\n')
      pos++;
  }

  return out;
}

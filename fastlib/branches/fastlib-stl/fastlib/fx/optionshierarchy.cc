#include "optionshierarchy.h"
#include "io.h"
#include "printing.h"
#include <iostream>

using namespace mlpack::io; 

/* Ctors, Dtors, and R2D2 [actually, just copy-tors] */
OptionsHierarchy::OptionsHierarchy() {
  nodeData.node = "";
  nodeData.desc = "";
  nodeData.tname = "";
  return;
}

OptionsHierarchy::OptionsHierarchy(const char* name) {
  nodeData.node = string(name);
  nodeData.desc = "";
  nodeData.tname = "";
  return;
}

OptionsHierarchy::~OptionsHierarchy() {
  return;
}

OptionsHierarchy::OptionsHierarchy(const OptionsHierarchy& other) {
  return;
}
  
/* Will never fail, as given paths are relative to current node
and will be generated if not found */
/* Also, we will insist on proper usage of C++ strings */
void OptionsHierarchy::AppendNode(string& pathname, string& tname) {
  string tmp = string("");
  OptionsData d;
  d.node = pathname;
  d.desc = tmp;
  d.tname = tname;
  AppendNode(pathname, tname, tmp, d);
}

void OptionsHierarchy::AppendNode(string& pathname, 
                                  string& tname, 
                                  string& description) {
  OptionsData d;
  d.node = pathname;
  d.desc = description; 
  d.tname = tname;
  AppendNode(pathname, tname, description, d);
}

void OptionsHierarchy::AppendNode(string& pathname, string& tname, 
                                  string& description, OptionsData& data) {
  string name = GetName(pathname);
  string path = GetPath(pathname);
  //Append the new name, if it isn't already there
  if (children.count(name) == 0)
    children[name] = OptionsHierarchy(name.c_str());
  
  if (pathname.find('/') == pathname.npos || path.length() < 1) {
    children[name].nodeData = data;
    return;
  }
  
  //Recurse until path is done
  children[name].AppendNode(path, tname, description, data);
}  

/* Returns the node with the specified pathname
  UNUSED, but lets keep it just in case...
OptionsHierarchy* OptionsHierarchy::findNode(string& pathname) {
  appendNode(pathname, ); //Make sure we actually have that node
  
  string name = getName(pathname);
  string path = getPath(pathname);
  
  //The node must be there, recurse to it.
  if(pathname.find('/') != pathname.npos)
    return children[name].findNode(path);

}*/

/* Returns the path in a pathname */
string OptionsHierarchy::GetPath(string& pathname) {
  //Want to make sure we return a valid string
  if (pathname.find('/') == pathname.npos)
    return string("");
  //Get the rest of the node name Eg foo/bar in root/foo/bar
  return pathname.substr(pathname.find('/')+1,pathname.length());
}

/* Returns the name in a pathname, eg foo in foo/bar/fizz */
string OptionsHierarchy::GetName(string& pathname) {
  //Want to makesure we return a valid string
  if (pathname.find('/') == pathname.npos)
    return pathname;
  //Get the topmost node name in this path Eg root in root/foo/bar
  return pathname.substr(0, pathname.find('/'));
}

void OptionsHierarchy::PrintAll() {
  PrintNode(); 

  map<string, OptionsHierarchy>::iterator iter;
  for (iter = children.begin(); iter != children.end(); iter++) {
    iter->second.PrintAll();
  }
}

void OptionsHierarchy::PrintNode() {
  cout << nodeData.node << ": ";
  Printing::PrintValue(nodeData.tname, nodeData.node);
  cout << endl;
}

void OptionsHierarchy::Print() {
  //Print the node, append '/' if that node is not a leaf
  PrintNode();
 
  //Begin formatted output 	
  cout << "Entries:" << endl;
  PrintLeaves();
  
  cout << "Submodules:" << endl;
  PrintBranches();
}

/* Prints all children nodes that have no children themselves */
void OptionsHierarchy::PrintLeaves() {
  map<string, OptionsHierarchy>::iterator iter;
  for (iter = children.begin(); iter != children.end(); iter++)
    if (!iter->second.children.size()) 
      //Print the node's name, data, and description.  
      iter->second.PrintNode(); 
      //Does it have a description?

}

/* Prints all children of this node which are parents */
void OptionsHierarchy::PrintBranches() {
  map<string, OptionsHierarchy>::iterator iter;
  
  //Iterate through all children
  for (iter = children.begin(); iter != children.end(); iter++)
  //Does this child have children?
    if (iter->second.children.size()) {
      iter->second.PrintNode();
    }
}

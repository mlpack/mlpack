#include "optionshierarchy.h"
#include <iostream>

/* Ctors, Dtors, and R2D2 [actually, just copy-tors] */
OptionsHierarchy::OptionsHierarchy() : node(""), isOutput(false) {
  return;
}

OptionsHierarchy::OptionsHierarchy(const char* name, bool isOut) : node(name), isOutput(isOut){
  node = name;
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
void OptionsHierarchy::appendNode(string& pathname) {
  string tmp = string("");
  appendNode(pathname, tmp);
}

void OptionsHierarchy::appendNode(string& pathname, string& description) {
  string name = getName(pathname);
  string path = getPath(pathname);
  
  //Append the new name, if it isn't already there
  if(children.count(name) == 0)
    children[name] = OptionsHierarchy(name.c_str());
  
  if(pathname.find('/') == pathname.npos || path.length() < 1) {
    children[name].desc = description;
    return;
  }
  
  //Recurse until path is done
  children[name].appendNode(path, description);
}  

/* Returns the node with the specified pathname */
OptionsHierarchy* OptionsHierarchy::findNode(string& pathname) {
  appendNode(pathname); //Make sure we actually have that node
  
  string name = getName(pathname);
  string path = getPath(pathname);
  
  //The node must be there, recurse to it.
  if(pathname.find('/') != pathname.npos)
    return children[name].findNode(path);
  else
    return this;
}

/* Returns the path in a pathname */
string OptionsHierarchy::getPath(string& pathname) {
  //Want to make sure we return a valid string
  if(pathname.find('/') == pathname.npos)
    return pathname;
  //Get the rest of the node name Eg foo/bar in root/foo/bar
  return pathname.substr(pathname.find('/')+1,pathname.length());
}

/* Returns the name in a pathname, eg foo in foo/bar/fizz */
string OptionsHierarchy::getName(string& pathname) {
  //Want to makesure we return a valid string
  if(pathname.find('/') == pathname.npos)
    return pathname;
  //Get the topmost node name in this path Eg root in root/foo/bar
  return pathname.substr(0, pathname.find('/'));
}


void OptionsHierarchy::print() {
  //Print the node, append '/' if that node is not a leaf
  std::cout << node << ":" << endl << endl;
  
  //Begin formatted output 	
  cout << "Entries:" << endl;
  printLeaves();
  
  cout << "Submodules:" << endl;
  printBranches();
}

void OptionsHierarchy::print(string& pathname) {
  //Get the topmost node name in this path Eg root in root/foo/bar
  string name = getName(pathname);
  //Get the rest of the node name Eg foo/bar in root/foo/bar
  string path = getPath(pathname);
  
  //Do we still need to recurse?
  if(path.length() > 0)
    if(children.count(name)) //Yes, and we can continue to do so
      return children[name].print(path);
    else if(node != name) { //Yes, but we can't find the next node.. are we already there?
      cout << "[!]\t Unknown Module: "<< pathname << endl;
      return;
    }
    
  print();
  
  return;
}


/* Prints all children nodes that have no children themselves */
void OptionsHierarchy::printLeaves() {
  map<string, OptionsHierarchy>::iterator iter;
  for(iter = children.begin(); iter != children.end(); iter++)
    if(!iter->second.children.size()) {
      cout << "\"" << iter->second.node << "\":" << endl;
      if(iter->second.desc.length() > 0)
        cout << "\t" << iter->second.desc << endl;
      else
        cout << "[!]\tUndocumented option" << endl;
    }
}

/* Prints all children of this node which are parents */
void OptionsHierarchy::printBranches() {
  map<string, OptionsHierarchy>::iterator iter;
  for(iter = children.begin(); iter != children.end(); iter++)
    if(iter->second.children.size()) {
      cout << "\"" << iter->second.node << "\":" << endl;
      if(iter->second.desc.length() > 0)
        cout << "\t" << iter->second.desc << endl;
      else
        cout << "[!]\tUndocumented module" << endl;
    }
}
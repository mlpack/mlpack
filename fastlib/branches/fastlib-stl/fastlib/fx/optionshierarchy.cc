#include "optionshierarchy.h"
#include "io.h"
#include "printing.h"
#include <iostream>

using namespace std;
using namespace mlpack::io; 

/* Ctors, Dtors, and R2D2 [actually, just copy-tors] */

/* Constructs an empty OptionsHierarchy node. */
OptionsHierarchy::OptionsHierarchy() {
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
OptionsHierarchy::OptionsHierarchy(const char* name) {
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
OptionsHierarchy::OptionsHierarchy(const OptionsHierarchy& other) {
  return;
}

/* 
 * Destroys the node.
 */
OptionsHierarchy::~OptionsHierarchy() {
  return;
}

/* 
 * Will never fail, as given paths are relative to current node
 * and will be generated if not found.
 * 
 * @param pathname The full pathname of the given node, eg /foo/bar.
 * @param tname A string unique to the type of the node.
 */
void OptionsHierarchy::AppendNode(string& pathname, string& tname) {
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
                                  string& description) {
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
/* 
 * Will return the node associated with a pathname 
 * 
 * @param pathname The full pathname of the node, 
 *   eg foo/bar in foo/bar.
 *
 * @return Pointer to the node with that pathname, 
 *   null if not found.
 */
OptionsHierarchy* OptionsHierarchy::FindNode(string& pathname) {
  return FindNodeHelper(pathname, pathname);  
}

OptionsHierarchy* OptionsHierarchy::FindNodeHelper(string& pathname, 
                                                  string& target) {
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
OptionsData OptionsHierarchy::GetNodeData() {
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
string OptionsHierarchy::GetPath(string& pathname) {
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
string OptionsHierarchy::GetName(string& pathname) {
  //Want to makesure we return a valid string
  if (pathname.find('/') == pathname.npos)
    return pathname;
  //Get the topmost node name in this path Eg root in root/foo/bar
  return pathname.substr(0, pathname.find('/'));
}

/*
 * Prints a node, followed by it's entries and submodules.
 */
void OptionsHierarchy::Print() {
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
void OptionsHierarchy::PrintAll() {
  PrintNode(); 
  map<string, OptionsHierarchy>::iterator iter;
  for (iter = children.begin(); iter != children.end(); iter++) {
    iter->second.PrintAll();
  }
}

/*
 * Prints every node and it's description.
 */
void OptionsHierarchy::PrintAllHelp() {
  PrintNodeHelp();
  map<string, OptionsHierarchy>::iterator iter;
  for (iter = children.begin(); iter != children.end(); iter++) {
    iter->second.PrintAllHelp();
  }
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


/* Prints all children nodes that have no children themselves */
void OptionsHierarchy::PrintLeaves() {
  map<string, OptionsHierarchy>::iterator iter;
  for (iter = children.begin(); iter != children.end(); iter++)
    if (!iter->second.children.size()) 
      //Print the node's name, data, and description.  
      iter->second.PrintNode(); 
      //Does it have a description?

}

/* 
 * Prints a node and it's value.
 */
void OptionsHierarchy::PrintNode() {
  cout << nodeData.node << ": " ;
  Printing::PrintValue(nodeData.tname, nodeData.node);
  cout << endl;
}

/* 
 * Prints a node and it's description.
 */
void OptionsHierarchy::PrintNodeHelp() {
  cout << nodeData.node << ": ";
  if (nodeData.desc.length() > 0)
    cout << endl << nodeData.desc;
  else
    cout << endl << "Undocumented module.";
  cout << endl << endl;
}


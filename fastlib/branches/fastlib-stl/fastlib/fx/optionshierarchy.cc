#include "optionshierarchy.h"
#include <iostream>

/* Ctors, Dtors, and R2D2 [actually, just copy-tors] */
OptionsHierarchy::OptionsHierarchy() {
	return;
}

OptionsHierarchy::OptionsHierarchy(const char* name) {
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
	//Get the topmost node name in this path Eg root in root/foo/bar
	string name = pathname.substr(0, pathname.find('/'));
	
	//Get the rest of the node name Eg foo/bar in root/foo/bar
	string path = pathname.substr(pathname.find('/')+1,pathname.length());
	
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

void OptionsHierarchy::print() {
	cout << node << ":" << std::endl;
	
	map<string, OptionsHierarchy>::iterator iter;
	for(iter = children.begin(); iter != children.end(); iter++)
		iter->second.print(1);
}

void OptionsHierarchy::print(int tabs) {
	for(int i = 0; i < tabs; i++)
		cout << "\t";
	
	//Print the node, append '/' if that node is not a leaf
	std::cout << node;
	if(children.size())
		cout << "/";
	if(desc.length())
		cout << "\t--" << desc; 
	cout << std::endl;
	
	map<string, OptionsHierarchy>::iterator iter;
	
	for(iter = children.begin(); iter != children.end(); iter++)
		iter->second.print(tabs+1);
}
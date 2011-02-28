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
void OptionsHierarchy::appendNode(string pathname) {
	//Get the topmost node name in this path Eg root in root/foo/bar
	string name = pathname.substr(0, pathname.find('/'));
	
	//Get the rest of the node name Eg foo/bar in root/foo/bar
	string path = pathname.substr(pathname.find('/')+1,pathname.length());
	
	//Append the new name, if it isn't already there
	if(children.count(name) == 0)
		children[name] = OptionsHierarchy(name.c_str());
	
	if(pathname.find('/') == pathname.npos)
		return;
	if(path.length() < 1)
		return;
	
	//Recurse until path is done
	children[name].appendNode(path);
}

void OptionsHierarchy::print() {
	std::cout << node << std::endl;
	
	std::map<string, OptionsHierarchy>::iterator iter;
	for(iter = children.begin(); iter != children.end(); iter++)
		iter->second.print(1);
}

void OptionsHierarchy::print(int tabs) {
	for(int i = 0; i < tabs; i++)
		std::cout << "\t";
	
	std::cout << node << std::endl;
	std::map<string, OptionsHierarchy>::iterator iter;
	
	for(iter = children.begin(); iter != children.end(); iter++)
		iter->second.print(tabs+1);
}
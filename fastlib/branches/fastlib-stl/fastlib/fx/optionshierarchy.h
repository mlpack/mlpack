#ifndef OPTIONSHIERARCHY_H
#define OPTIONSHIERARCHY_H

#include <map>
#include <string>

using namespace std; //We'll be using std a lot in this file

class OptionsHierarchy {
	private:
		/* Name of this node */
		string node;
		/* Description of this node, if any */
		string desc;
		/* Map of this node's children.  All children should have a
			corresponding OptionsHierarchy, hence the references */
		map<string, OptionsHierarchy> children;
	
	public:
		/* Ctors, Dtors, and R2D2 [actually, just copy-tors] */
		OptionsHierarchy();
		OptionsHierarchy(const char* name);
		~OptionsHierarchy();
		OptionsHierarchy(const OptionsHierarchy& other);
	
		/* Will never fail, as given paths are relative to current node
		and will be generated if not found */
		/* Also, we will insist on proper usage of C++ strings */
		void appendNode(string& pathname);
		void appendNode(string& pathname, string& description);
	
		/* Print functions */
		void print();
		void print(string& pathname);
		void printLeaves();
		void printBranches();

};

#endif
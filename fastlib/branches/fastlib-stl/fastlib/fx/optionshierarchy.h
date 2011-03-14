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
    /* Should this node be printed upon a printOutput event? */
    bool isOutput; 
  
  /* Returns the name foo in the pathname foo/bar/fizz */
  string getName(string& pathname);
  /* Returns the path bar/fizz in the pathname foo/bar/fizz */
  string getPath(string& pathname);
  
  public:
    /* Ctors, Dtors, and R2D2 [actually, just copy-tors] */
    OptionsHierarchy();
    OptionsHierarchy(const char* name, bool isOut = false);
    OptionsHierarchy(const OptionsHierarchy& other);
    virtual ~OptionsHierarchy();
  
    /* Will never fail, as given paths are relative to current node
    and will be generated if not found */
    /* Also, we will insist on proper usage of C++ strings */
    void appendNode(string& pathname);
    void appendNode(string& pathname, string& description);
  
    /* Will return the node associated with a pathname */
    OptionsHierarchy* findNode(string& pathname);

    /* Print functions */
    void print();
    void print(string& pathname);
    void printLeaves();
    void printBranches();
  
    /* Prints only nodes which are registered as outputs. */
    void printOutputs();
};

#endif
#ifndef OPTIONSHIERARCHY_H
#define OPTIONSHIERARCHY_H

#include <map>
#include <string>

using namespace std; //We'll be using std a lot in this file

/* Aids in the extensibility of OptionsHierarchy by focusing the potential changes into 
  one structure. */
struct OptionsData {
  /* Name of this node */
  string node;
  /* Description of this node, if any */
  string desc;
  /* Type information of this node */
  string tname;
};

class OptionsHierarchy {
  private:
    /* Holds all node specific data */
    OptionsData nodeData;

    /* Map of this node's children.  All children should have a
      corresponding OptionsHierarchy, hence the references */
    map<string, OptionsHierarchy> children;
  
    /* Returns the name foo in the pathname foo/bar/fizz */
    string getName(string& pathname);
    /* Returns the path bar/fizz in the pathname foo/bar/fizz */
    string getPath(string& pathname);
  
  public:
    /* Ctors, Dtors, and R2D2 [actually, just copy-tors] */
    OptionsHierarchy();
    OptionsHierarchy(const char* name);
    OptionsHierarchy(const OptionsHierarchy& other);
    virtual ~OptionsHierarchy();
  
    /* Will never fail, as given paths are relative to current node
    and will be generated if not found */
    /* Also, we will insist on proper usage of C++ strings */
    void appendNode(string& pathname, string& tname);
    void appendNode(string& pathname, string& tname, string& description);
    void appendNode(string& pathname, string& tname, string& description, OptionsData& data);
  
    /* Will return the node associated with a pathname */
    //OptionsHierarchy* findNode(string& pathname);

    /* Print functions */
    void print();
    void printAll();
    void printNode();
    void printLeaves();
    void printBranches();
};

#endif

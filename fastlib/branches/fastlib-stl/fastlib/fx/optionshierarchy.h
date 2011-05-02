#ifndef OPTIONSHIERARCHY_H
#define OPTIONSHIERARCHY_H

#include <map>
#include <string>

namespace mlpack {
namespace io {

/* Aids in the extensibility of OptionsHierarchy by 
   focusing the potential changes into one structure. */
struct OptionsData {
  /* Name of this node */
  std::string node;
  /* Description of this node, if any */
  std::string desc;
  /* Type information of this node */
  std::string tname;
};


class OptionsHierarchy {
  private:
   /* Holds all node specific data */
   OptionsData nodeData;

   /* Map of this node's children.  All children should have a
      corresponding OptionsHierarchy, hence the references */
   typedef std::map<std::string, OptionsHierarchy> ChildMap;
   ChildMap children;
  
   /* Returns the name foo in the pathname foo/bar/fizz */
   std::string GetName(std::string& pathname);
   /* Returns the path bar/fizz in the pathname foo/bar/fizz */
   std::string GetPath(std::string& pathname);
  
  public:
   /* Ctors, Dtors, and R2D2 [actually, just copy-tors] */
   OptionsHierarchy();
   OptionsHierarchy(const char* name);
   OptionsHierarchy(const OptionsHierarchy& other);
   virtual ~OptionsHierarchy();
  
   /* Will never fail, as given paths are relative to current node
      and will be generated if not found */
   /* Also, we will insist on proper usage of C++ strings */
   void AppendNode(std::string& pathname, std::string& tname);
   void AppendNode(std::string& pathname, std::string& tname,
                   std::string& description);
   void AppendNode(std::string& pathname, std::string& tname, 
                   std::string& description, OptionsData& data);
  
   /* Will return the node associated with a pathname */
   void FindNode(std::string& pathname);
   void FindNodeHelper(std::string& pathname, std::string& target);

   /* Print functions */
   //Prints a single node, and outlines relations
   void Print();

   //Prints all nodes, plus their data
   void PrintAll();
   //Prints a node and its data
   void PrintNode();

   //Prints all nodes, plus their description
   void PrintAllHelp();
   //Prints a node and its description
   void PrintNodeHelp();

   //Prints the leaves of a node
   void PrintLeaves();
   //Prints the branches of a node
   void PrintBranches();
};

}; // namespace io
}; // namespace mlpack
#endif

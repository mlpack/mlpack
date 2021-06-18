#include "print_py_wrapper.hpp"
#include "get_methods_wrapper.hpp"
#include "get_class_name_wrapper.hpp"
#include "get_program_name.hpp"

#include <mlpack/core/util/io.hpp>

using namespace mlpack::util;
using namespace std;

namespace mlpack {
namespace bindings {
namespace python {

void PrintWrapperPY(const std::vector<std::string>& groupProgramNames,
										const std::string& groupName,
				    			  const std::string& validMethods)
{
	map<string, map<string, ParamData>> accumulate; // this to store all parameters in a group.

	vector<string> methods = GetMethods(validMethods);

	string importString = "";

	for(int i=0; i<methods.size(); i++)
	{
		importString += "from " + groupName + "_" + methods[i] + " ";
		importString += "import " + groupName + "_" + methods[i] + "\n";
		IO::RestoreSettings(groupProgramNames[i]);
		accumulate[methods[i]] = IO::Parameters();
		IO::ClearSettings();
	}

	cout << importString << endl;

	string className = groupName; // create class name from group name

	// print class.
	cout << "class " << className << ":" << endl;
	cout << "  def __init__():" << endl;
	cout << "    pass" << endl;

	typedef map<string, ParamData>::iterator ParamItr;
	for(int i=0; i<methods.size(); i++)
	{
		cout << "  " << "def " << methods[i] << "(" << endl;
		int indent = 5 /* ' def ' */ + methods[i].size() + 1 /* '(' */;
		for(ParamItr itr=accumulate[methods[i]].begin();
				itr != accumulate[methods[i]].end(); ++itr)
		{
			if(itr->first == "lambda")
				cout << string(indent, ' ') << "lambda_" << "," << endl;
			else
				cout << string(indent, ' ') << itr->first << "," << endl;
		}
		cout << string(indent, ' ') << "):" << endl;
		cout << "  " << "  pass" << endl;
	}
}

}
}
}

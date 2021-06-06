#include "print_py_wrapper.hpp"
#include "get_methods.hpp"
#include "get_class_name.hpp"
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
	map<string, map<string, ParamData>> accumulate;

	vector<string> methods = GetMethods(validMethods);
	typedef vector<string>::iterator MethodItr;

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

	vector<string> methodSpecificParams;
	typedef map<string, ParamData>::iterator ParamItr;
	string className = GetClassName(groupName);

	cout << "class " << className << ":" << endl;
}

}
}
}

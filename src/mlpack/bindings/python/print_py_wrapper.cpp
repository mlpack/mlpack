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
	typedef std::map<std::string, util::ParamData>::iterator ParamIter;
	for(auto& name : groupProgramNames)
	{
		IO::RestoreSettings(name);
		std::map<std::string, util::ParamData>& parameters = IO::Parameters();
		for (ParamIter it = parameters.begin(); it != parameters.end(); ++it)
		{
			cout << it->first << endl;
		}
		IO::ClearSettings();
	}

	vector<string> methods = GetMethods(validMethods);

	string importString = "";

	for(auto& method : methods)
	{
		importString += "from " + groupName + "_" + method + " ";
		importString += "import " + groupName + "_" + method + "\n";
	}

	cout << importString << endl;

	string className = GetClassName(groupName);

	cout << "class " << className << ":" << endl;
}

}
}
}

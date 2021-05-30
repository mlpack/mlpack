#include "print_py_wrapper.hpp"
#include "get_methods.hpp"
#include "get_class_name.hpp"
#include <mlpack/core/util/io.hpp>

using namespace mlpack::util;
using namespace std;

namespace mlpack {
namespace bindings {
namespace python {

void PrintWrapperPY(const std::string& groupName,
										const std::string& validMethods)
{
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

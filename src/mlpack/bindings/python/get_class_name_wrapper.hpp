#ifndef MLPACK_BINDINGS_PYTHON_GET_CLASS_NAME_WRAPPER_HPP
#define MLPACK_BINDINGS_PYTHON_GET_CLASS_NAME_WRAPPER_HPP

#include <string>
#include <sstream>

using namespace std;

namespace mlpack {
namespace bindings {
namespace python {

string GetClassName(const string& groupName)
{
	string className = "";
	stringstream groupNameStream(groupName);
	string temp;

	while(getline(groupNameStream, temp, '_'))
	{
		temp[0] = toupper(temp[0]);
		className += temp;
	}

	return className;
}

} // python
} // bindings
} // mlpack

#endif

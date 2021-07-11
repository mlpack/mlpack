#ifndef MLPACK_BINDINGS_PYTHON_GET_PROGRAM_NAME_HPP
#define MLPACK_BINDINGS_PYTHON_GET_PROGRAM_NAME_HPP

#include <string>
#include <sstream>

using namespace std;

namespace mlpack {
namespace bindings {
namespace python {

string GetProgramName(const string& groupName, const string& method)
{
	string programName = "";
	stringstream groupNameStream(groupName);
	string temp;

	while(getline(groupNameStream, temp, '_'))
	{
		temp[0] = toupper(temp[0]);
		programName += temp + " ";
	}

	temp = method;
	temp[0] = toupper(temp[0]);

	programName += temp;

	return programName;
}

}
}
}

#endif

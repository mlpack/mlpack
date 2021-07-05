#ifndef MLPACK_BINDINGS_PYTHON_GET_METHODS_HPP
#define MLPACK_BINDINGS_PYTHON_GET_METHODS_HPP

#include <vector>
#include <string>
#include <sstream>

using namespace std;

namespace mlpack {
namespace bindings {
namespace python {

vector<string> GetMethods(const string& validMethods)
{
	vector<string> methods;
	stringstream methodStream(validMethods);
	string temp;

	while(getline(methodStream, temp, ' '))
	{
		methods.push_back(temp);
	}

	return methods;
}

} // python
} // bindings
} // mlpack

#endif

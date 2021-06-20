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
	set<string> serializable; // to store list of all serializable types.

	typedef set<string>::iterator SerialItr;
	typedef map<string, ParamData>::iterator ParamItr;

	vector<string> methods = GetMethods(validMethods);

	string importString = "";

	for(int i=0; i<methods.size(); i++)
	{
		importString += "from mlpack." + groupName + "_" + methods[i] + " ";
		importString += "import " + groupName + "_" + methods[i] + "\n";
		IO::RestoreSettings(groupProgramNames[i]);
		accumulate[methods[i]] = IO::Parameters();
		bool temp_val;
		for(ParamItr itr=IO::Parameters().begin(); itr!=IO::Parameters().end(); ++itr)
		{
			IO::GetSingleton().functionMap[itr->second.tname]["IsSerializable"](itr->second, NULL, (void*)& temp_val);
			if(temp_val)
				serializable.insert(itr->second.cppType);
		}
		IO::ClearSettings();
	}

	cout << importString << endl;

	string className = GetClassName(groupName); // create class name from group name

	// print class.
	cout << "class " << className << ":" << endl;
	cout << "  def __init__(self):" << endl;
	for(SerialItr itr = serializable.begin(); itr != serializable.end(); ++itr)
	{
		cout << "    self." << *itr << " = None" << endl;
	}
	cout << endl;

	typedef map<string, ParamData>::iterator ParamItr;
	for(int i=0; i<methods.size(); i++)
	{
		IO::RestoreSettings(groupProgramNames[i]);

		cout << "  " << "def " << methods[i] << "(self," << endl;
		int indent = 5 /* ' def ' */ + methods[i].size() + 2 /* '(' */;

		// printing arguments of a method.
		for(ParamItr itr=accumulate[methods[i]].begin();
				itr != accumulate[methods[i]].end(); ++itr)
		{
			if(itr->second.input && (serializable.find(itr->second.cppType) == serializable.end()))
			{
				// if(itr->first == "lambda")
				// 	cout << string(indent, ' ') << "lambda_" << "," << endl;
				// else
				// 	cout << string(indent, ' ') << itr->first << "," << endl;
				ParamData& d = accumulate[methods[i]][itr->first];
				cout << string(indent, ' ');
				IO::GetSingleton().functionMap[d.tname]["PrintDefn"](d, NULL, NULL);
				cout << "," << endl;
			}
		}
		cout << string(indent-1, ' ') << "):" << endl;

		// calling internal functions.
		cout << "  " << "  out=" << groupName << "_" << methods[i] << "(" << endl;
		indent = 4 + 4 /* out = */ + groupName.length() + 1 /*_*/ + methods[i].size() + 1 /*(*/;
		for(ParamItr itr=accumulate[methods[i]].begin();
				itr != accumulate[methods[i]].end(); ++itr)
		{
			if(itr->second.input)
			{
				if((serializable.find(itr->second.cppType) == serializable.end()))
				{
					if(itr->first == "lambda")
						cout << string(indent, ' ') << "lambda_" << "=" << "lambda_" << "," << endl;
					else
						cout << string(indent, ' ') << itr->first << "=" << itr->first << "," << endl;
				}
				else
				{
					cout << string(indent, ' ') << itr->first << "=" << "self." << itr->second.cppType << "," << endl;
				}
			}
		}
		cout << string(indent-1, ' ') << ")" << endl;

		// returning the output parameters and store serializable output params.
		for(ParamItr itr=accumulate[methods[i]].begin();
				itr != accumulate[methods[i]].end(); ++itr)
		{
			if(!itr->second.input)
			{
				if((serializable.find(itr->second.cppType) != serializable.end()))
					cout << "    " << "self." << itr->second.cppType << "=" << "out[\"" << itr->first << "\"]" << endl;
				cout << "    " << "return out[\"" << itr->first << "\"]" << endl;
				break; // only return one output parameter.
			}
		}
		cout << endl;
		IO::ClearSettings();
	}
}

}
}
}

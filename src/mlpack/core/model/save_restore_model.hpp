#ifndef SAVE_RESTORE_MODEL_HPP
#define SAVE_RESTORE_MODEL_HPP

#include <err.h>
#include <list>
#include <map>
#include <sstream>
#include <string>

#include <libxml/parser.h>
#include <libxml/tree.h>

#include <armadillo>
#include <boost/tokenizer.hpp>

#include "model.hpp"

namespace mlpack
{
  namespace model
  {
    class SaveRestoreModel : public Model
    {
      private:
        std::map<std::string, std::string> parameters;

      public:
        SaveRestoreModel() {}
        ~SaveRestoreModel() { parameters.clear(); }
        bool readFile (std::string filename);
        void recurseOnNodes (xmlNode* n);
        bool writeFile (std::string filename);
        template<typename T>
        T& loadParameter (T& t, std::string name);
        char loadParameter (char c, std::string name);
        arma::mat& loadParameter (arma::mat& matrix, std::string name);
        template<typename T>
        void saveParameter (T& t, std::string name);
        void saveParameter (char c, std::string name);
        void saveParameter (arma::mat& mat, std::string name);
        virtual bool loadModel (std::string filename)
        {
          return true;
        }
        virtual bool saveModel (std::string filename)
        {
          return true;
        }
    };
  };
};

#include "save_restore_model_impl.hpp"

#endif

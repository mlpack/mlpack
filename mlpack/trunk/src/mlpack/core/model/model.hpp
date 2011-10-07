#ifndef MODEL_HPP
#define MODEL_HPP

namespace mlpack
{
  namespace model
  {

    class Model
    {
      public:
        Model () {}
        virtual ~Model () {}
        virtual bool solve () = 0;
    };
  };
};

#endif

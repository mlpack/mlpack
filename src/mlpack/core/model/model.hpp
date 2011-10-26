#ifndef __MLPACK_CORE_MODEL_MODEL_HPP
#define __MLPACK_CORE_MODEL_MODEL_HPP

namespace mlpack {
namespace model {

class Model
{
  public:
    Model () {}
    virtual ~Model () {}
    virtual bool solve () = 0;
};

}; // namespace model
}; // namespace mlpack

#endif // __MLPACK_CORE_MODEL_MODEL_HPP

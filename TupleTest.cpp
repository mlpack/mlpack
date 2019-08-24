#include <iostream>
#include <tuple>

size_t v1;
size_t v2;

size_t& getV1() { return v1; }

std::tuple<size_t&, size_t&> getVal()
{
    return std::forward_as_tuple(v1, v2);
}
int main()
{
    std::tuple<size_t&, size_t&> t = getVal();
    std::cout << std::get<0>(t) << '\t' << std::get<1>(t) << '\n';
    std::get<0>(t) = 4;
    std::cout << v1 << '\t' << v2 << '\n'; // I want v1 to be 4 here
}

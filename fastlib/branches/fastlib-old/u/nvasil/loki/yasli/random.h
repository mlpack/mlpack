#ifndef YASLI_RANDOM_H_
#define YASLI_RANDOM_H_

// $Id: random.h 754 2006-10-17 19:59:11Z syntheticpp $


#include <ctime>

class Random 
{
    unsigned int seed_;
public:
    Random(unsigned int seed = 0)
        : seed_(seed ? seed : static_cast<unsigned int>(std::time(0)))
    {
    }
    unsigned short nextShort()
    {
        /* Use any number from this list for "a"
            18000 18030 18273 18513 18879 19074 19098 19164 19215 19584       
            19599 19950 20088 20508 20544 20664 20814 20970 21153 21243       
            21423 21723 21954 22125 22188 22293 22860 22938 22965 22974       
            23109 23124 23163 23208 23508 23520 23553 23658 23865 24114       
            24219 24660 24699 24864 24948 25023 25308 25443 26004 26088       
            26154 26550 26679 26838 27183 27258 27753 27795 27810 27834       
            27960 28320 28380 28689 28710 28794 28854 28959 28980 29013       
            29379 29889 30135 30345 30459 30714 30903 30963 31059 31083
        */
        static const unsigned int a = 18000;
        return static_cast<unsigned short>(seed_ = 
            a * (seed_ & 65535) + 
                (seed_ >> 16));
    }

    unsigned int nextUint()
    {
        return (unsigned int)nextShort() << (CHAR_BIT * sizeof(unsigned short)) |
            nextShort();
    }
    unsigned int nextUint(unsigned int high)
    {
        assert(high < ULONG_MAX - 1);
        ++high;
        const unsigned int bucket_size = ULONG_MAX / high;
        unsigned int a;
        do 
        {
            a = nextUint() / bucket_size;
        }
        while (a >= high);
        return a;
    }
};

#endif // RANDOM_H_


namespace math__private {
  template<int t_numerator, int t_denominator = 1>
  struct ZPowImpl {
    static double Calculate(double d) {
      return pow(d, t_numerator * 1.0 / t_denominator);
    }
  };

  template<int t_equal>
  struct ZPowImpl<t_equal, t_equal> {
    static double Calculate(double d) {
      return d;
    }
  };

  template<>
  struct ZPowImpl<1, 1> {
    static double Calculate(double d) {
      return d;
    }
  };

  template<>
  struct ZPowImpl<1, 2> {
    static double Calculate(double d) {
      return sqrt(d);
    }
  };

  template<>
  struct ZPowImpl<1, 3> {
    static double Calculate(double d) {
      return cbrt(d);
    }
  };

  template<int t_denominator>
  struct ZPowImpl<0, t_denominator> {
    static double Calculate(double d) {
      return 1;
    }
  };

  template<int t_numerator>
  struct ZPowImpl<t_numerator, 1> {
    static double Calculate(double d) {
      return ZPowImpl<t_numerator - 1, 1>::Calculate(d) * d;
    }
  };

  // absolute-value-power: have special implementations for even powers
  // TODO: do this for all even integer powers

  template<int t_numerator, int t_denominator, bool is_even>
  struct ZPowAbsImpl;

  template<int t_numerator, int t_denominator>
  struct ZPowAbsImpl<t_numerator, t_denominator, false> {
    static double Calculate(double d) {
      return ZPowImpl<t_numerator, t_denominator>::Calculate(fabs(d));
    }
  };

  template<int t_numerator, int t_denominator>
  struct ZPowAbsImpl<t_numerator, t_denominator, true> {
    static double Calculate(double d) {
      return ZPowImpl<t_numerator, t_denominator>::Calculate(d);
    }
  };
};

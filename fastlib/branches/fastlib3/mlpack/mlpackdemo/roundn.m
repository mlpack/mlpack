function [x,msg] = roundn(x,n)

%ROUNDN  Rounds input data at specified power of 10
%
%  y = ROUNDN(x) rounds the input data x to the nearest hundredth.
%
%  y = ROUNDN(x,n) rounds the input data x at the specified power
%  of tens position.  For example, n = -2 rounds the input data to
%  the 10E-2 (hundredths) position.
%
%  [y,msg] = ROUNDN(...) returns the text of any error condition
%  encountered in the output variable msg.
%
%  See also ROUND

%  Copyright 1996-2002 Systems Planning and Analysis, Inc. and The MathWorks, Inc.
%  Written by:  E. Byrns, E. Brown
%   $Revision: 1.9 $    $Date: 2002/03/20 21:26:19 $

msg = [];   %  Initialize output

if nargin == 0
    error('Incorrect number of arguments')
elseif nargin == 1
    n = -2;
end

%  Test for scalar n

if max(size(n)) ~= 1
   msg = 'Scalar accuracy required';
   if nargout < 2;  error(msg);  end
   return
elseif ~isreal(n)
   warning('Imaginary part of complex N argument ignored')
   n = real(n);
end

%  Compute the exponential factors for rounding at specified
%  power of 10.  Ensure that n is an integer.

factors  = 10 ^ (fix(-n));

%  Set the significant digits for the input data

x = round(x * factors) / factors;

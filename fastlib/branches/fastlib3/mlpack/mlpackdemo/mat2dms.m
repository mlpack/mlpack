function dmsvec = mat2dms(d,m,s,n)

%MAT2DMS Converts a [deg min sec] matrix to vector format
%
%  dms = MAT2DMS(d,m,s) converts a deg:min:sec matrix into a vector
%  format.  The vector format is dms = 100*deg + min + sec/100.
%  This allows d,m,s triple to be compressed into a single value,
%  which can then be employed similar to a degree or radian vector.
%  The inputs d, m and s must be of equal size.  Minutes and
%  second must be between 0 and 60.
%
%  dms = MAT2DMS(mat) assumes and input matrix of [d m s].  This is
%  useful only for single column vectors for d, m and s.
%
%  dms = MAT2DMS(d,m) and dms = MAT2DMS([d m]) assume that seconds
%  are zero, s = 0.
%
%  dms = MAT2DMS(d,m,s,n) uses n as the accuracy of the seconds
%  calculation.  n = -2 uses accuracy in the hundredths position,
%  n = 0 uses accuracy in the units position.  Default is n = -5.
%  For further discussion of the input n, see ROUNDN.
%
%  See also DMS2MAT

%  Copyright 1996-2002 Systems Planning and Analysis, Inc. and The MathWorks, Inc.
%  Written by:  E. Byrns, E. Brown
%   $Revision: 1.10 $    $Date: 2002/03/20 21:25:51 $


if nargin == 0
   error('Incorrect number of arguments')

elseif nargin==1
   if size(d,2)== 3
       s = d(:,3);   m = d(:,2);   d = d(:,1);
   elseif size(d,2)== 2
       m = d(:,2);   d = d(:,1);   s = zeros(size(d));
   elseif size(d,2) == 0
       d = [];   m = [];   s = [];
   else
       error('Single input matrices must be n-by-2 or n-by-3.');
   end
   n = -5;

elseif nargin == 2
   s = zeros(size(d));
   n = -5;

elseif nargin == 3
   n = -5;
end

%  Test for empty arguments

if isempty(d) & isempty(m) & isempty(s);  dmsvec = [];  return;  end

%  Don't let seconds be rounded beyond the tens place.
%  If you did, then 55 seconds rounds to 100, which is not good.

if n == 2;  n = 1;   end

%  Complex argument tests

if any([~isreal(d) ~isreal(m) ~isreal(s)])
    warning('Imaginary parts of complex ANGLE argument ignored')
	d = real(d);   m = real(m);   s = real(s);
end

%  Dimension and value tests

if ~isequal(size(d),size(m),size(s))
    error('Inconsistent dimensions for input arguments')
elseif any(rem(d(~isnan(d)),1) ~= 0 | rem(m(~isnan(m)),1) ~= 0)
    error('Degrees and minutes must be integers')
end

if any(abs(m) > 60) | any (abs(m) < 0)       %  Actually algorithm allows for
    error('Minutes must be >= 0 and < 60')   %  up to exactly 60 seconds or
                                             %  60 minutes, but the error message
elseif any(abs(s) > 60) | any(abs(s) < 0)    %  doesn't reflect this so that angst
    error('Seconds must be >= 0 and < 60')   %  is minimized in the user docs
end

%  Ensure that only one negative sign is present and at the correct location

if any((s<0 & m<0) | (s<0 & d<0) | (m<0 & d<0) )
    error('Multiple negative entries in a DMS specification')
elseif any((s<0 & (m~=0 | d~= 0)) | (m<0 & d~=0))
    error('Incorrect negative DMS specification')
end

%  Construct a sign vector which has +1 when
%  angle >= 0 and -1 when angle < 0.  Note that the sign of the
%  angle is associated with the largest nonzero component of d:m:s

negvec = (d<0) | (m<0) | (s<0);
signvec = ~negvec - negvec;

%  Convert to all positive numbers.  Allows for easier
%  adjusting at 60 seconds and 60 minutes

d = abs(d);     m = abs(m);    s = abs(s);

%  Truncate seconds to a specified accuracy to eliminate round-off errors

[s,msg] = roundn(s,n);
if ~isempty(msg);   error(msg);   end

%  Adjust for 60 seconds or 60 minutes. If s > 60, this can only be
%  from round-off during roundn since s > 60 is already tested above.
%  This round-off effect has happened though.

indx = find(s >= 60);
if ~isempty(indx);   m(indx) = m(indx) + 1;   s(indx) = 0;   end

%  The user can not put minutes > 60 as input.  However, the line
%  above may create minutes > 60 (since the user can put in m == 60),
%  thus, the test below includes the greater than condition.

indx = find(m >= 60);
if ~isempty(indx);   d(indx) = d(indx) + 1;   m(indx) = m(indx)-60;   end

%  Construct the dms vector format

dmsvec = signvec .* (100*d + m + s/100);

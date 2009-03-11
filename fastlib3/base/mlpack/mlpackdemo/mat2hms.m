function hmsvec = mat2hms(h,m,s,n)

%MAT2HMS Converts a [hrs min sec] matrix to vector format
%
%  hms = MAT2HMS(h,m,s) converts a hrs:min:sec matrix into a vector
%  format.  The vector format is hms = 100*hrs + min + sec/100.
%  This allows h,m,s triple to be compressed into a single value,
%  which can then be employed similar to a second or hour vector.
%  The inputs h, m and s must be of equal size.  Minutes and
%  second must be between 0 and 60.
%
%  hms = MAT2HMS(mat) assumes and input matrix of [h m s].  This is
%  useful only for single column vector for h, m and s.
%
%  hms = MAT2HMS(h,m) and hms = MAT2HMS([h m]) assume that seconds
%  are zero, s = 0.
%
%  hms = MAT2HMS(h,m,s,n) uses n as the accuracy of the seconds
%  calculation.  n = -2 uses accuracy in the hundredths position,
%  n = 0 uses accuracy in the units position.  Default is n = -5.
%  For further discussion of the input n, see ROUNDN.
%
%  See also HMS2MAT

%  Copyright 1996-2002 Systems Planning and Analysis, Inc. and The MathWorks, Inc.
%  Written by:  E. Byrns, E. Brown
%  $Revision: 1.10 $    $Date: 2002/03/20 21:25:51 $

if nargin == 0
   error('Incorrect number of arguments')

elseif nargin==1
   if size(h,2)== 3
       s = h(:,3);   m = h(:,2);   h = h(:,1);
   elseif size(h,2)== 2
       m = h(:,2);   h = h(:,1);   s = zeros(size(h));
   elseif size(h,2) == 0
       h = [];   m = [];   s = [];
   else
       error('Single input matrices must be n-by-2 or n-by-3.');
   end
   n = -5;

elseif nargin == 2
   s = zeros(size(h));
   n = -5;

elseif nargin == 3
   n = -5;
end

%  Test for empty arguments

if isempty(h) & isempty(m) & isempty(s);  hmsvec = [];  return;  end

%  Don't let seconds be rounded beyond the tens place.
%  If you did, then 55 seconds rounds to 100, which is not good.

if n == 2;  n = 1;   end

%  Complex argument tests

if any([~isreal(h) ~isreal(m) ~isreal(s)])
    warning('Imaginary parts of complex TIME argument ignored')
	h = real(h);   m = real(m);   s = real(s);
end

%  Dimension and value tests

if  ~isequal(size(h),size(m),size(s))
    error('Inconsistent dimensions for input arguments')
elseif any(rem(h(~isnan(h)),1) ~= 0 | rem(m(~isnan(m)),1) ~= 0)
    error('Hours and minutes must be integers')
end

if any(abs(m) > 60) | any (abs(m) < 0)       %  Actually algorithm allows for
    error('Minutes must be >= 0 and < 60')   %  up to exactly 60 seconds or
                                             %  60 minutes, but the error message
elseif any(abs(s) > 60) | any(abs(s) < 0)    %  doesn't reflect this so that angst
    error('Seconds must be >= 0 and < 60')   %  is minimized in the user docs
end

%  Ensure that only one negative sign is present

if any((s<0 & m<0) | (s<0 & h<0) | (m<0 & h<0) )
    error('Multiple negative entries in a hms specification')
elseif any((s<0 & (m~=0 | h~= 0)) | (m<0 & h~=0))
    error('Incorrect negative HMS specification')
end

%  Construct a sign vector which has +1 when
%  time >= 0 and -1 when time < 0.  Note that the sign of the
%  time is associated with the largest nonzero component of h:m:s

negvec = (h<0) | (m<0) | (s<0);
signvec = ~negvec - negvec;

%  Convert to all positive numbers.  Allows for easier
%  adjusting at 60 seconds and 60 minutes

h = abs(h);     m = abs(m);    s = abs(s);

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
if ~isempty(indx);   h(indx) = h(indx) + 1;   m(indx) = m(indx)-60;   end

%  Construct the hms vector format

hmsvec = signvec .* (100*h + m + s/100);

function [hout,mout,sout] = hms2mat(hms,n)

%HMS2MAT Converts a hms vector format to a [hrs min sec] matrix
%
%  [h,m,s] = HMS2MAT(hms) converts a hms vector format to a
%  hrs:min:sec matrix.  The vector format is hms = 100*hrs + min + sec/100.
%  This allows compressed hms data to be expanded to a h,m,s triple,
%  for easier reporting and viewing of the data.
%
%  [h,m,s] = HMS2MAT(hms,n) uses n digits in the accuracy of the
%  seconds calculation.  n = -2 uses accuracy in the hundredths position,
%  n = 0 uses accuracy in the units position.  Default is n = -5.
%  For further discussion of the input n, see ROUNDN.
%
%  mat = HMS2MAT(...) returns a single output argument of mat = [h m s].
%  This is useful only if the input hms is a single column vector.
%
%       See also MAT2HMS

%  Copyright 1996-2002 Systems Planning and Analysis, Inc. and The MathWorks, Inc.
%  Written by:  E. Byrns, E. Brown
%  $Revision: 1.9 $    $Date: 2002/03/20 21:25:36 $

if nargin == 0
     error('Incorrect number of arguments')
elseif nargin == 1
     n = -5;
end

%  Test for empty arguments

if isempty(hms); hout = []; mout = []; sout = []; return; end

%  Test for complex arguments

if ~isreal(hms)
     warning('Imaginary parts of complex TIME argument ignored')
	 hms = real(hms);
end

%  Don't let seconds be rounded beyond the tens place.
%  If you did, then 55 seconds rounds to 100, which is not good.

if n == 2;  n = 1;   end

%  Construct a sign vector which has +1 when hms >= 0 and -1 when hms < 0.

signvec = sign(hms);
signvec = signvec + (signvec == 0);   %  Ensure +1 when hms = 0

%  Decompress the hms data vector

hms = abs(hms);
h = fix(hms/100);                      %  Hours
m = fix(hms) - abs(100*h);             %  Minutes
[s,msg] = roundn(100*rem(hms,1),n);    %  Seconds:  Truncate to roundoff error
if ~isempty(msg);   error(msg);   end

%  Adjust for 60 seconds or 60 minutes.
%  Test for seconds > 60 to allow for round-off from roundn,
%  Test for minutes > 60 as a ripple effect from seconds > 60


indx = find(s >= 60);
if ~isempty(indx);   m(indx) = m(indx) + 1;   s(indx) = s(indx) - 60;   end
indx = find(m >= 60);
if ~isempty(indx);   h(indx) = h(indx) + 1;   m(indx) =  m(indx) - 60;   end

%  Data consistency checks

if any(m > 59) | any (m < 0)
    error('Minutes must be >= 0 and <= 59')

elseif any(s >= 60) | any( s < 0)
    error('Seconds must be >= 0 and < 60')
end

%  Determine where to store the sign of the time.  It should be
%  associated with the largest nonzero component of h:m:s.

hsign = signvec .* (h~=0);
msign = signvec .* (h==0 & m~=0);
ssign = signvec .* (h==0 & m==0 & s~=0);

%  In the application of signs below, the ~ operator is used so that
%  the sign vector contains only +1 and -1.  Any zero occurances causes
%  data to be lost when the sign has been applied to a higher component
%  of h:m:s.  Use fix function to eliminate potential round-off errors.

h = (~hsign + hsign).*fix(h);      %  Apply signs to the hours
m = (~msign + msign).*fix(m);      %  Apply signs to minutes
s = (~ssign + ssign).*s;           %  Apply signs to seconds


%  Set the output arguments

if nargout <= 1
    hout = [h m s];
elseif nargout == 3
    hout = h;   mout = m;   sout = s;
else
    error('Invalid number of output arguments')
end

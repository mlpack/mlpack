function hms=hr2hms(hrs)

%HR2HMS Converts time from hours to hrs:min:sec vector format
%
%  hms = HR2HMS(hr) converts time from hours to hrs:min:sec
%  vector format.
%
%  See also HMS2HR,  HR2SEC, MAT2HMS, HMS2MAT, TIMEDIM, TIME2STR

%  Copyright 1996-2002 Systems Planning and Analysis, Inc. and The MathWorks, Inc.
%  Written by:  E. Byrns, E. Brown
%  $Revision: 1.9 $    $Date: 2002/03/20 21:25:38 $

if nargin==0
	error('Incorrect number of arguments')
elseif ~isreal(hrs)
     warning('Imaginary parts of complex TIME argument ignored')
     hrs = real(hrs);
end

%  Test for empty inputs

if isempty(hrs);     hms = [];   return;   end

%  Construct a sign vector which has +1 when hrs >= 0 and -1 when hrs < 0.

signvec = sign(hrs);
signvec = signvec + (signvec == 0);    %  Enforce +1 when hrs == 0

%  Compute the hours, minutes and seconds

hrs = abs(hrs);           %  Work in absolute value.  Signvec will set sign later
h   = fix(hrs);           %  Hours
ms  = 60*(hrs - h);       %  Minutes and seconds
m   = fix(ms);            %  Minutes
s   = 60*(ms - m);        %  Seconds

%  Determine where to store the sign of the time.  It should be
%  associated with the largest nonzero component of h:m:s.

hsign = signvec .* (h~=0);                %  Associate with hours
msign = signvec .* (h==0 & m~=0);         %  Assoicate with minutes (h = 0)
ssign = signvec .* (h==0 & m==0 & s~=0);  %  Associate with seconds (h = m = 0)

%  In the application of signs below, the ~ operator is used so that
%  the sign vector contains only +1 and -1.  Any zero occurances causes
%  data to be lost when the sign has been applied to a higher component
%  of h:m:s.

h = (~hsign + hsign).*h;      %  Apply signs to the hours
m = (~msign + msign).*m;      %  Apply signs to minutes
s = (~ssign + ssign).*s;      %  Apply signs to seconds


hms = mat2hms(h,m,s);     %  Construct the hms vector for output

function hm=hr2hm(hr)

%HR2HM Converts time from hours to hrs:min format
%
%  hm = HR2HM(hr) converts times from hours to hrs:min vector format.
%
%  See also SEC2HM, HR2HMS, TIMEDIM, TIME2STR

%  Copyright 1996-2002 Systems Planning and Analysis, Inc. and The MathWorks, Inc.
%  Written by:  E. Brown, E. Byrns
%  $Revision: 1.9 $    $Date: 2002/03/20 21:25:38 $

if nargin==0;   error('Incorrect number of arguments');   end

%  Compute the time in hm.
%  0.2 is used to round seconds.  0.3+0.2 = 0.5 which will round up
%  to an additional minute.  0.29+0.2 = 0.49 which will stay at
%  the curren minute.

hms = round(hr2hms(hr)+0.2);
[h,m,s] = hms2mat(hms);
hm = mat2hms(h,m);         %  Round 60 minutes to 1 hour here

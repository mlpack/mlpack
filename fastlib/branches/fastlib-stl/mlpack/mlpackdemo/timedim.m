function timemat = timedim(timemat,from,to)

%TIMEDIM  Converts times from one unit system to another
%
%  t = TIMEDIM(tin,'from','to') converts times between
%  recognized units.  Input and output units are entered as strings.
%  This function allows access to all time conversions based upon input
%  unit strings.  Allowable units strings are:  'hm' for hms:min;
%  'hms' for hms:min:sec; 'hours' or 'hr' for hours;
%  'seconds' or 'sec' for seconds.
%
%  See also HMS2HR, HMS2SEC, HMS2HM, HR2SEC, HR2HMS, HR2HM,
%          SEC2HMS,  SEC2HR, SEC2HM

%  Copyright 1996-2002 Systems Planning and Analysis, Inc. and The MathWorks, Inc.
%  Written by:  E. Brown, E. Byrns
%  $Revision: 1.10 $    $Date: 2002/03/20 21:26:32 $

if nargin ~= 3;    error('Incorrect number of arguments');    end


[from,msg] = unitstr(from,'time');    %  Test the input strings for recognized units
if ~isempty(msg);   error(msg);  end

[to,msg] = unitstr(to,'time');        %  Return the full name in lower case
if ~isempty(msg);   error(msg);  end

%  Complex input test

if ~isreal(timemat)
     warning('Imaginary parts of complex TIME argument ignored')
     timemat = real(timemat);
end

%  If no unit changes, then simply return

if strcmp(from,to);	  return;    end

%  Find the appropriate string matches and transform the angles

switch from          %  Switch statment faster that if/elseif
    case 'hours'
	      switch to
				case 'hm',        timemat = hr2hm(timemat);
				case 'hms',       timemat = hr2hms(timemat);
		        case 'seconds',   timemat = hr2sec(timemat);
				otherwise,        error('Unrecognized time units string')
	     end

    case 'hm'
	      switch to
				case 'hours',     timemat = hms2hr(timemat);
				case 'hms',       timemat = timemat;
		        case 'seconds',   timemat = hms2sec(timemat);
				otherwise,        error('Unrecognized time units string')
	     end

    case 'hms'
	      switch to
				case 'hours',     timemat = hms2hr(timemat);
				case 'hm',        timemat = hms2hm(timemat);
		        case 'seconds',   timemat = hms2sec(timemat);
				otherwise,        error('Unrecognized time units string')
	     end

    case 'seconds'
	      switch to
				case 'hours',     timemat = sec2hr(timemat);
				case 'hm',        timemat = sec2hm(timemat);
		        case 'hms',       timemat = sec2hms(timemat);
				otherwise,        error('Unrecognized time units string')
	     end

	otherwise
          error('Unrecognized time units string')
end


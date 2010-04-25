function [str,msg] = unitstr(str0,measstr)

%UNITSTR  Tests for valid unit string or abbreviations
%
%  UNITSTR displays a list of recognized unit string in
%  the Mapping Toolbox.
%
%  str = UNITSTR('str0','measstr') tests for valid unit strings or
%  abbreviations.  If a string or abbreviation is found, then the output
%  string is set to the corresponding preset string.
%
%  The second input to determine the measurement system to be used.
%  Allowable strings are 'angles' for angle unit checks;  'distances'
%  for distance unit checks; and 'time' for time unit checks.
%
%  [str,msg] = UNITSTR(...) returns the string describing any error
%  condition encountered.
%
%  See also ANGLEDIM, DISTDIM, TIMEDIM

%  Copyright 1996-2002 Systems Planning and Analysis, Inc. and The MathWorks, Inc.
%  Written by:  E. Byrns, E. Brown
%  $Revision: 1.12 $  $Date: 2002/03/20 21:26:35 $

%  Input argument tests

if nargin == 0;
     unitstra;  unitstrd;  unitstrt;  return
elseif nargin ~= 2
    error('Incorrect number of arguments')
end

%  Initialize outputs

str = [];   msg = [];

%  Test for a valid measurement string only if it is not already
%  an exact match.  This approach is faster that simply always
%  using strmatch.

measstr = lower(measstr);
if ~strcmp(measstr,'angles') & ~strcmp(measstr,'distances') & ...
   ~strcmp(measstr,'times')
       validparm = ['angles   ';  'distances'; 'times    '];
       indx = strmatch(lower(measstr),validparm);
       if length(indx) == 1;    measstr = deblank(validparm(indx,:));
           else;                error('Unrecognized MEASUREMENT string')
       end
end

%  Test for a valid units string in the appropriate measurement units

switch measstr
     case 'angles',      [str,msg] = unitstra(str0);
	 case 'distances',   [str,msg] = unitstrd(str0);
	 case 'times',       [str,msg] = unitstrt(str0);
end

if ~isempty(msg)
    if nargout ~= 2;   error(msg);   end
end


%************************************************************************
%************************************************************************
%************************************************************************


function [str,msg] = unitstra(str0)

%UNITSTRA  Tests for valid angle unit string or abbreviations
%
%  Purpose
%
%  UNITSTRA tests for valid angle unit strings or abbreviations.
%  If a valid string or abbreviation is found, then the
%  unit string is set to a preset string.  This allows
%  users to enter strings in a variety of formats, but then
%  this function determines the standard format for each
%  string.  This allows other functions to work with the
%  standard format unit strings.
%
%  Synopsis
%
%       unitstra              %  Displays a list of recongized strings
%       str = unitstra(str0)  %  Produces a standard format unit string
%       [str,errmsg] = unitstra(str0)
%            If two output arguments are supplied, then error condition
%            messages are returned to the calling function for processing.
%
%       See also UNITSTRD, UNITSTRT, ANGLEDIM


%  Define unit string names only if necessary.  This process
%  takes some time and should be done only if it is truly necessary.
%  This martix definition process is significantly faster than strvcat,
%  where the padding must be computed

%  Initialize outputs

str = [];   msg = [];

%  Display list or set input strings to lower case for comparison

if nargin == 0
      units = ['degrees'; 'dm     '; 'dms    '; 'radians'];
      abbreviations = ['deg   for degrees'
		               'rad   for radians'];
	 disp(' ');    disp('Recognized Angle Unit Strings')
	 disp(' ');    disp(units)
	 disp(' ');    disp('Recognized Angle Unit Abbreviations')
	 disp(' ');    disp(abbreviations)
	 return
elseif ~isstr(str0)
     msg = 'Input argument must be a string';
	 if nargout < 2;  error(msg);  end
	 return
else
    str0 = lower(str0);
end

%  Test for an appropriate string word.
%  Test for exact matches because this is a faster procedure that
%  the strmatch function.

switch str0
    case 'degrees',      str = str0;        return
    case 'deg',          str = 'degrees';   return
    case 'radians',      str = str0;        return
    case 'rad',          str = 'radians';   return
    case 'dm',           str = str0;        return
    case 'dms',          str = str0;        return
    otherwise,
           units = ['degrees'; 'dm     '; 'dms    '; 'radians'];
           strindx = strmatch(str0,units);
end

%  Set the output string or error message

if length(strindx) == 1
     str = deblank(units(strindx,:));   %  Set the name string
else
     msg = ['Unrecognized angle units string:  ',str0];
	 if nargout < 2;  error(msg);  end
	 return
end


%************************************************************************
%************************************************************************
%************************************************************************


function [str,msg] = unitstrd(str0)

%UNITSTRD  Tests for valid distance unit string or abbreviations
%
%  Purpose
%
%  UNITSTRD tests for valid distance unit strings or abbreviations.
%  If a valid string or abbreviation is found, then the
%  unit string is set to a preset string.  This allows
%  users to enter strings in a variety of formats, but then
%  this function determines the standard format for each
%  string.  This allows other functions to work with the
%  standard format unit strings.
%
%  Synopsis
%
%       unitstrd              %  Displays a list of recongized strings
%       str = unitstrd(str0)  %  Produces a standard format unit string
%       [str,errmsg] = unitstrd(str0)
%            If two output arguments are supplied, then error condition
%            messages are returned to the calling function for processing.
%
%       See also UNITSTRA, UNITSTRT, DISTDIM


%  Define unit string names only if necessary.  This process
%  takes some time and should be done only if it is truly necessary.
%  This martix definition process is significantly faster than strvcat,
%  where the padding must be computed

%  Initialize outputs

str = [];   msg = [];

%  Display list or set input strings to lower case for comparison

if nargin == 0
      units = ['degrees      '; 'feet         '; 
	  		   'kilometers   '; 'kilometres   ';
	 		   'meters       '; 'metres       '; 'nauticalmiles';
	           'radians      '; 'statutemiles '];

      abbreviations = ['deg          for degrees       '
	                   'ft           for feet          '
	                   'km           for kilometers    '
	                   'm            for meters        '
	                   'mi or miles  for statute miles '
		               'nm           for nautical miles'
		               'rad          for radians       '
		               'sm           for statute miles ' ];
	 disp(' ');    disp('Recognized Distance Unit Strings')
	 disp(' ');    disp(units)
	 disp(' ');    disp('Recognized Distance Abbreviations')
	 disp(' ');    disp(abbreviations)
	 return
elseif ~isstr(str0)
     msg = 'Input argument must be a string';
	 if nargout < 2;  error(msg);  end
	 return
else
    str0 = lower(str0);
end

%  Test for a valid abbreviation or appropriate string word.
%  Test for exact matches because this is a faster procedure that
%  the strmatch function.

switch str0
    case 'deg',                str = 'degrees';        return
    case 'km',                 str = 'kilometers';     return
    case 'm',                  str = 'meters';         return
    case 'mi',                 str = 'statutemiles';   return
    case 'miles',              str = 'statutemiles';   return
    case 'nm',                 str = 'nauticalmiles';  return
    case 'sm',                 str = 'statutemiles';   return
    case 'ft',                 str = 'feet';           return
    case 'degrees',            str = str0;             return
    case 'kilometers',         str = str0;             return
    case 'kilometres',         str = 'kilometers';     return
    case 'meters',             str = str0;             return
    case 'metres',             str = 'meters';         return
    case 'feet',               str = str0;             return
    case 'foot',               str = 'feet';           return
    case 'nauticalmiles',      str = str0;             return
    case 'radians',            str = str0;             return
    case 'statutemiles',       str = str0;             return
    otherwise,
           units = ['degrees      '; 'feet         '; 
		   			'kilometers   '; 'meters       '; 'nauticalmiles'
	                'radians      '; 'statutemiles '];
           strindx = strmatch(str0,units);
end

%  Set the output string or error message

if length(strindx) == 1
     str = deblank(units(strindx,:));   %  Set the name string

else
     msg = ['Unrecognized distance units string:  ',str0];
	 if nargout < 2;  error(msg);  end
	 return
end


%************************************************************************
%************************************************************************
%************************************************************************


function [str,msg] = unitstrt(str0)

%UNITSTRT  Tests for valid time unit string or abbreviations
%
%  Purpose
%
%  UNITSTRT tests for valid time unit strings or abbreviations.
%  If a valid string or abbreviation is found, then the
%  unit string is set to a preset string.  This allows
%  users to enter strings in a variety of formats, but then
%  this function determines the standard format for each
%  string.  This allows other functions to work with the
%  standard format unit strings.
%
%  Synopsis
%
%       unitstrt              %  Displays a list of recongized strings
%       str = unitsttr(str0)  %  Produces a standard format unit string
%       [str,errmsg] = unitstrt(str0)
%            If two output arguments are supplied, then error condition
%            messages are returned to the calling function for processing.
%
%       See also UNITSTRA, UNITSTRD, TIMEDIM


%  Define unit string names only if necessary.  This process
%  takes some time and should be done only if it is truly necessary.
%  This martix definition process is significantly faster than strvcat,
%  where the padding must be computed

%  Initialize outputs

str = [];   msg = [];

%  Display list or set input strings to lower case for comparison

if nargin == 0
     units = ['hm     '; 'hms    '; 'hours  '; 'seconds'];
     abbreviations = ['hr     for hours  '; 'sec    for seconds'];
	 disp(' ');    disp('Recognized Time Unit Strings')
	 disp(' ');    disp(units)
	 disp(' ');    disp('Recognized Time Abbreviations')
	 disp(' ');    disp(abbreviations)
	 return
elseif ~isstr(str0)
     msg = 'Input argument must be a string';
	 if nargout < 2;  error(msg);  end
	 return
else
    str0 = lower(str0);
end

%  Test for a valid abbreviation or appropriate string word.
%  Test for exact matches because this is a faster procedure that
%  the strmatch function.

switch str0
    case 'hr',                  str = 'hours';        return
    case 'sec',                 str = 'seconds';      return
    case 'hm',                  str = str0;           return
    case 'hms',                 str = str0;           return
    case 'hours',               str = str0;           return
    case 'seconds',             str = str0;           return
    otherwise,
           units = ['hm     '; 'hms    '; 'hours  '; 'seconds'];
           strindx = strmatch(str0,units);
end

%  Set the output string or error message

if length(strindx) == 1
     str = deblank(units(strindx,:));   %  Set the name string

else
     msg = ['Unrecognized time units string:  ',str0];
	 if nargout < 2;  error(msg);  end
	 return
end


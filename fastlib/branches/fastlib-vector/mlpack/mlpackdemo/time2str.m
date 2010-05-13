function strout = time2str(timein,clock,format,units,digits)

%TIME2STR  Time conversion to a string
%
%  str = TIME2STR(t) converts a numerical vector of times to
%  a string matrix.  The output string matrix is useful for the
%  display of times.
%
%  str = TIME2STR(t,'clock') uses the specified clock input to
%  construct the string matrix.  Allowable clock strings are
%  '12' for a regular 12 hour clock (AM and PM);  '24' for a
%  regular 24 hour clock; and 'nav' for a navigational hour clock.
%  If omitted or blank, '24' is assumed.
%
%  str = TIME2STR(t,'clock','format') uses the specified format input
%  to construct the string matrix.  Allowable format strings are
%  'hm' for hours and minutes; and 'hms' for hours, minutes
%  and seconds.  If omitted or blank, 'hm' is assumed.
%
%  str = TIME2STR(t,'clock','format','units') defines the units which
%  the input times are supplied.  Any valid time units string can be
%  entered.  If omitted or blank, 'hours' is assumed.
%
%  str = TIME2STR(t,'clock','format',digits) uses the input digits to
%  determine the number of decimal digits in the output matrix.
%  n = -2 uses accuracy in the hundredths position, n = 0 uses
%  accuracy in the units position.  Default is n = 0.  For further
%  discussion of the input n, see ROUNDN.
%
%  str = TIME2STR(t,'clock','format','units',digits) uses all the inputs
%  to construct the output string matrix.
%
%  See also TIMEDIM

%  Copyright 1996-2002 Systems Planning and Analysis, Inc. and The MathWorks, Inc.
%  Written by:  E. Byrns, E. Brown
%   $Revision: 1.10 $    $Date: 2002/03/20 21:26:32 $


if nargin == 0
    error('Incorrect number of arguments')

elseif nargin == 1
    clock = [];    format = [];     units  = [];     digits = [];

elseif nargin == 2
    format = [];     units  = [];     digits = [];

elseif nargin == 3
    units  = [];     digits = [];

elseif nargin == 4
    if isstr(units)
	    digits = [];
	else
	    digits = units;   units  = [];
	end
end

%  Empty argument tests

if isempty(digits);  digits  = 0;   end

if isempty(format);           format = 'hms';
    elseif ~isstr(format);    error('FORMAT input must be a string')
    else;                     format = lower(format);
end
if isempty(clock);            clock = '24';
    elseif isstr(clock);      clock = lower(clock);
    else;                     clock = num2str(clock);
end

if isempty(units);        units  = 'hours';
    else;                 [units,msg]  = unitstr(units,'time');
                          if ~isempty(msg);   error(msg);   end
end

%  Test the format string

if ~strcmp(format,'hm') & ~strcmp(format,'hms')
    error('Unrecognized format string')
end

%  Prevent complex times

if ~isreal(timein)
    warning('Imaginary parts of complex TIME argument ignored')
	timein = real(timein);
end

%  Ensure that inputs are a column vector

timein = timein(:);

%  Compute the time (h,m,s) in 24 hour increments.
%  Eliminate 24 hour days from the time string.

timein = timedim(timein,units,'hours');
timein = timedim(rem(timein,24),'hours',format);
[h,m,s] = hms2mat(timein,digits);

%  Work with positive (h,m,s).  Prefix character takes care of
%  signs separately

h = abs(h);   m = abs(m);    s = abs(s);

%  Compute the prefix and suffix matrices.
%  Note that the * character forces a space in the output

prefix = ' ';     prefix = prefix(ones(size(timein)),:);
indx = find(timein<0);  if ~isempty(indx);   prefix(indx) = '-';   end

switch clock
   case '12'
      suffix = '* M';      suffix = suffix(ones(size(timein)),:);
      indx = find(h<12);   if ~isempty(indx); suffix(indx,2) = 'A';   end
      indx = find(h>=12);  if ~isempty(indx); suffix(indx,2) = 'P'; end
      indx = find(h>12);   if ~isempty(indx); h(indx) = h(indx) - 12; end

   otherwise
      suffix = ' ';     suffix = suffix(ones(size(timein)),:);
end

%  Compute the time string

if strcmp(clock(1),'n')  %  Navigational clock
      fillstr = ' ';     fillstr = fillstr( ones(size(timein)) );

      h_str = num2str(h,'%02g');     %  Convert hours to a string

%  Determine the round to increments 15 seconds

      indx1 = find(s<7.5);
	  indx2 = find(s >= 7.55 & s < 22.5);
	  indx3 = find(s >= 22.5 & s < 37.5);
	  indx4 = find(s >= 37.5 & s < 52.5);
	  indx5 = find(s >= 52.5);

      s_str = zeros([length(h) 3]);   %  Blank seconds string

	  if ~isempty(indx1);  s_str(indx1,1:3) = '***';     end
	  if ~isempty(indx2);  s_str(indx2,1:3) = '''**';    end
	  if ~isempty(indx3);  s_str(indx3,1:3) = '''''*';   end
	  if ~isempty(indx4);  s_str(indx4,1:3) = '''''''';  end
	  if ~isempty(indx5);  s_str(indx5,1:3) = '***';  m(indx5) = m(indx5)+1;  end

%  Convert minutes now because an increment of m may have occurred

      m_str = num2str(m,'%02g');     %  Convert minutes to a string

else        %  12 or 24 hour regular clock
      fillstr = ':';     fillstr = fillstr( ones(size(timein)) );

%  Construct the format string for converting seconds

	  rightdigits  = abs(min(digits,0));
      if rightdigits > 0;   totaldigits = 3+ rightdigits;
	      else              totaldigits = 2+ rightdigits;
	  end
	  formatstr = ['%0',num2str(totaldigits),'.',num2str(rightdigits),'f'];

%  Hours, minutes and seconds

      h_str = num2str(h,'%02g');       %  Convert hours to a string
      m_str = num2str(m,'%02g');       %  Convert minutes to a string
      s_str = num2str(s,formatstr);    %  Convert seconds to a padded string
end

%  Construct the display string

if strcmp(format,'hms')
    strout = [prefix  h_str fillstr  m_str fillstr  s_str suffix];
else
    strout = [prefix  h_str fillstr  m_str suffix];
end

%  Right justify each row of the output matrix.  This places
%  all extra spaces in the leading position.  Then strip these
%  lead zeros.  Left justifying and then a DEBLANK call will
%  not ensure that all strings line up.  LEADBLNK only strips
%  leading blanks which appear in all rows of a string matrix,
%  thereby not messing up any right justification of the string matrix.

strout = shiftspc(strout);
strout = leadblnk(strout,' ');

%  Replace the hold characters with a space

indx = find(strout == '*');
if ~isempty(indx);  strout(indx) = ' ';  end

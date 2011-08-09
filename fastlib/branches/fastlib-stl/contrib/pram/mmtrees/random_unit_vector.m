function an=random_unit_vector(varargin)
% random_unit_vector
% random_unit_vector(n)
% random_unit_vector(m,n)
% random_unit_vector([m n])
% random_unit_vector('double')
% random_unit_vector(n,'double')
% random_unit_vector(m,n,'double')
% random_unit_vector([m n],'double')
% random_unit_vector('single')
% random_unit_vector(n,'single')
% random_unit_vector(m,n,'single')
% random_unit_vector([m n],'single')

% m - dimentionarity 
% n - number of unit vectors

md=3; % default m

isg=false; % if single
switch nargin
    case 0
        % if no inputs
        n=1;
        m=md;
    case 1
        %
        i1=varargin{1};
        if ischar(i1)
            if strcmpi(i1,'single')
                isg=true;
            end
        else
            if length(i1)==1
                m=md;
                n=i1;
            else
                m=i1(1);
                n=i1(2);
            end
        end
    case 2
        i1=varargin{1};
        i2=varargin{2};
        if ischar(i2)
            if length(i1)==1
                m=md;
                n=i1;
            else
                m=i1(1);
                n=i1(2);
            end
            if strcmpi(i2,'single')
                isg=true;
            end
        else
            m=i1;
            n=i2;
        end
    case 3
        m=varargin{1};
        n=varargin{2};
        if strcmpi(varargin{3},'single')
            isg=true;
        end
            
        
end

% simple case of 1d
if m==1
    if isg
        an=single(2*(randn(1,n)>0)-1);
    else
        an=2*(randn(1,n)>0)-1;
    end
    return;
end

if isg
    v=randn(m,n,'single');
else
    v=randn(m,n);
end

% normalize:
if isg
    an=zeros(m,n,'single');
else
    an=zeros(m,n);
end
for nc=1:n
    while 1
        v2=v(:,nc)'*v(:,nc);
        if v2>1e-10 % too small values must be excluded 
            % because it will have discretization errors
            an(:,nc)=v(:,nc)/sqrt(v2);
            break;
        else
            % otherwise repeat random generation
            if isg
                v(:,nc)=randn(m,1,'single');
            else
                v(:,nc)=randn(m,1);
            end
        end
    end
end


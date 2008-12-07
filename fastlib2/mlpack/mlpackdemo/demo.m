% global fastlib_root =' ';

function varargout = demo(varargin)
% DEMO_EXPORT M-file for demo_export.fig
%      DEMO_EXPORT, by itself, creates a new DEMO_EXPORT or raises the existing
%      singleton*.
%
%      H = DEMO_EXPORT returns the handle to a new DEMO_EXPORT or the handle to
%      the existing singleton*.
%
%      DEMO_EXPORT('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DEMO_EXPORT.M with the given input arguments.
%
%      DEMO_EXPORT('Property','Value',...) creates a new DEMO_EXPORT or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before demo_export_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to demo_export_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help demo_export

% Last Modified by GUIDE v2.5 30-Nov-2008 22:37:20

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @demo_export_OpeningFcn, ...
                   'gui_OutputFcn',  @demo_export_OutputFcn, ...
                   'gui_LayoutFcn',  @demo_export_LayoutFcn, ...
                   'gui_Callback',   []);
if nargin & isstr(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before demo_export is made visible.
function demo_export_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to demo_export (see VARARGIN)

% Choose default command line output for demo_export
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% This sets up the initial plot - only do when we are invisible
% so window can get raised using demo_export.
if strcmp(get(hObject,'Visible'),'off')
    plot(rand(5));
end

% UIWAIT makes demo_export wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = demo_export_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.axes1);
cla;

popup_sel_index = get(handles.popupmenu1, 'Value');
switch popup_sel_index
    case 1
        plot(rand(5));
    case 2
        plot(sin(1:0.01:25));
    case 3
        comet(cos(1:.01:10));
    case 4
        bar(1:10);
    case 5
        plot(membrane);
    case 6
        surf(peaks);
end


% --------------------------------------------------------------------
function FileMenu_Callback(hObject, eventdata, handles)
% hObject    handle to FileMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function OpenMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to OpenMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = uigetfile('*.fig');
if ~isequal(file, 0)
    open(file);
end

% --------------------------------------------------------------------
function PrintMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to PrintMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
printdlg(handles.figure1)

% --------------------------------------------------------------------
function CloseMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to CloseMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
selection = questdlg(['Close ' get(handles.figure1,'Name') '?'],...
                     ['Close ' get(handles.figure1,'Name') '...'],...
                     'Yes','No','Yes');
if strcmp(selection,'No')
    return;
end

delete(handles.figure1)


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

set(hObject, 'String', {'plot(rand(5))', 'plot(sin(1:0.01:25))', 'comet(cos(1:.01:10))', 'bar(1:10)', 'plot(membrane)', 'surf(peaks)'});

% --- Executes on selection change in popupmenu3.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns popupmenu3 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu3


% --- Executes during object creation, after setting all properties.
function axis_CreateFcn(hObject, eventdata, handles)
% hObject    handle to dim_reduction (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
hold off;
plot(0);
hold on;
set(hObject, 'Tag', 'first_axis');
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function axis1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to dim_reduction (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
hold off;
plot(0);
hold on;
set(hObject, 'Tag', 'second_axis');
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function dim_reduction_CreateFcn(hObject, eventdata, handles)
% hObject    handle to dim_reduction (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

% --- Executes on selection change in dim_reduction.
function dim_reduction_Callback(hObject, eventdata, handles)
% hObject    handle to dim_reduction (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns dim_reduction contents as cell array
%        contents{get(hObject,'Value')} returns selected item from dim_reduction
contents = get(hObject, 'String');
handles.dim_reduction_var = contents{get(hObject, 'Value')};
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function neighbors_CreateFcn(hObject, eventdata, handles)
% hObject    handle to neighbors (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end


% --- Executes on selection change in neighbors.
function neighbors_Callback(hObject, eventdata, handles)
% hObject    handle to neighbors (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns neighbors contents as cell array
%        contents{get(hObject,'Value')} returns selected item from neighbors
contents = get(hObject, 'String');
handles.neighbors_var = contents{get(hObject, 'Value')};
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function algorithms_CreateFcn(hObject, eventdata, handles)
% hObject    handle to algorithms (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end


% --- Executes on selection change in algorithms.
function algorithms_Callback(hObject, eventdata, handles)
% hObject    handle to algorithms (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns algorithms contents as cell array
%        contents{get(hObject,'Value')} returns selected item from algorithms
contents = get(hObject, 'String');
handles.algorithms_var = contents{get(hObject, 'Value')};
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function data_file1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to data_file1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function data_file1_Callback(hObject, eventdata, handles)
% hObject    handle to data_file1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of data_file1 as text
%        str2double(get(hObject,'String')) returns contents of data_file1 as a double
contents = get(hObject, 'String');
handles.data_file_var = contents{get(hObject, 'Value')};
guidata(hObject, handles);

% --- Executes on button press in plot.
function store_range_Callback(hObject, eventdata, handles)
% hObject    handle to plot (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
tmp_lower_vector = [handles.range11_var ; handles.range21_var];
tmp_upper_vector = [handles.range12_var ; handles.range22_var];
handles.lower_ranges = [handles.lower_ranges tmp_lower_vector];
handles.upper_ranges = [handles.upper_ranges tmp_upper_vector];
% Draw the rectangles as well.
axes(handles.first_axis);
cla;
data_file_list = get(handles.data_file1, 'String');
data_matrix = load(data_file_list{get(handles.data_file1, 'Value')});
plot(data_matrix(:, 1), data_matrix(:, 2), '.');
for i = 1:size(handles.lower_ranges, 2)
    width = handles.upper_ranges(1, i) - handles.lower_ranges(1, i);
    height = handles.upper_ranges(2, i) - handles.lower_ranges(2, i);
    rectangle('Position', [handles.lower_ranges(1, i) ...
        handles.lower_ranges(2, i) width height]);
    drawnow;
end;
guidata(hObject, handles);
zoom on;


% --- Executes on button press in go1.
function go1_Callback(hObject, eventdata, handles)
% hObject    handle to go1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
data_file_list = get(handles.data_file1, 'String');
data_file_var = data_file_list{get(handles.data_file1, 'Value')};
dim_reduction_list = get(handles.dim_reduction1, 'String');
dim_reduction_var = dim_reduction_list{get(handles.dim_reduction1, 'Value')};
ComputeAlgorithms(data_file_var, dim_reduction_var,...
    hObject, handles);
% Add the dimension-reduced dataset to the list.

% --- Executes on button press in go2.
function go2_Callback(hObject, eventdata, handles)
% hObject    handle to go2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
data_file_list = get(handles.data_file1, 'String');
data_file_var = data_file_list{get(handles.data_file1, 'Value')};
neighbors_list = get(handles.neighbors1, 'String');
neighbors_var = neighbors_list{get(handles.neighbors1, 'Value')};
ComputeAlgorithms(data_file_var, neighbors_var,...
    hObject, handles);
% Get the handle to the plot in the GUI and plot adjacnecy graph.
data_file_list = get(handles.data_file1, 'String');
data_file_var = data_file_list{get(handles.data_file1, 'Value')};
data_matrix = load(data_file_var);
% Dimension reduce to two dimensions, in case it has more than two.
[U, S, V] = svd(data_matrix, 'econ');
U = U(:, 1:2);
S = S(1:2, 1:2);
data_matrix = U * S;
load 'adjacency_matrix.mat' adjacency_matrix;
axes(handles.first_axis);
cla;
gplot(adjacency_matrix, data_matrix,'.-');
set(get(gca,'Children'), 'MarkerEdgeColor', [1 0 0])
set(get(gca,'Children'), 'MarkerSize', 5)
zoom on;

% --- Executes on button press in go3.
function go3_Callback(hObject, eventdata, handles)
% hObject    handle to go3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
data_file_list = get(handles.data_file1, 'String');
data_file_var = data_file_list{get(handles.data_file1, 'Value')};
algorithms_list = get(handles.algorithms1, 'String');
algorithms_var = algorithms_list{get(handles.algorithms1, 'Value')};
ComputeAlgorithms(data_file_var, algorithms_var,...
    hObject, handles);

% --- Executes during object creation, after setting all properties.
function knn1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to knn1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function knn1_Callback(hObject, eventdata, handles)
% hObject    handle to knn1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of knn1 as text
%        str2double(get(hObject,'String')) returns contents of knn1 as a double
handles.knn1_var = str2num(get(hObject, 'String'));
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function bandwidth1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to bandwidth1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function bandwidth1_Callback(hObject, eventdata, handles)
% hObject    handle to knn1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of bandwidth1 as text
%        str2double(get(hObject,'String')) returns contents of bandwidth1
%        as a double

% --- Executes during object creation, after setting all properties.
function range11_CreateFcn(hObject, eventdata, handles)
% hObject    handle to bandwidth1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end


function range11_Callback(hObject, eventdata, handles)
% hObject    handle to knn1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of bandwidth1 as text
%        str2double(get(hObject,'String')) returns contents of bandwidth1 as a double
handles.range11_var = str2num(get(hObject, 'String'));
if isfield(handles, 'lower_ranges') == 0
    handles.lower_ranges = [];
end;
if isfield(handles, 'upper_ranges') == 0
    handles.upper_ranges = [];
end;
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function range12_CreateFcn(hObject, eventdata, handles)
% hObject    handle to bandwidth1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function range12_Callback(hObject, eventdata, handles)
% hObject    handle to knn1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of bandwidth1 as text
%        str2double(get(hObject,'String')) returns contents of bandwidth1 as a double
handles.range12_var = str2num(get(hObject, 'String'));
if isfield(handles, 'lower_ranges') == 0
    handles.lower_ranges = [];
end;
if isfield(handles, 'upper_ranges') == 0
    handles.upper_ranges = [];
end;
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function range21_CreateFcn(hObject, eventdata, handles)
% hObject    handle to bandwidth1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function range21_Callback(hObject, eventdata, handles)
% hObject    handle to knn1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of bandwidth1 as text
%        str2double(get(hObject,'String')) returns contents of bandwidth1 as a double
handles.range21_var = str2num(get(hObject, 'String'));
if isfield(handles, 'lower_ranges') == 0
    handles.lower_ranges = [];
end;
if isfield(handles, 'upper_ranges') == 0
    handles.upper_ranges = [];
end;
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function range22_CreateFcn(hObject, eventdata, handles)
% hObject    handle to bandwidth1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function range22_Callback(hObject, eventdata, handles)
% hObject    handle to knn1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of bandwidth1 as text
%        str2double(get(hObject,'String')) returns contents of bandwidth1 as a double
handles.range22_var = str2num(get(hObject, 'String'));
if isfield(handles, 'lower_ranges') == 0
    handles.lower_ranges = [];
end;
if isfield(handles, 'upper_ranges') == 0
    handles.upper_ranges = [];
end;
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function textbox_output_CreateFcn(hObject, eventdata, handles)
% hObject    handle to data_file1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end




function textbox_output_Callback(hObject, eventdata, handles)
% hObject    handle to data_file1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of data_file1 as text
%        str2double(get(hObject,'String')) returns contents of data_file1 as a double


% --- Creates and returns a handle to the GUI figure. 
function h1 =demo_export_LayoutFcn(policy)
% policy - create a new figure or use a singleton. 'new' or 'reuse'.

persistent hsingleton;
if strcmpi(policy, 'reuse') & ishandle(hsingleton)
    h1 = hsingleton;
    return;
end

h1 = figure(...
'Units','normalized',...
'Color',[0.831372549019608 0.815686274509804 0.784313725490196],...
'Colormap',[0 0 0.5625;0 0 0.625;0 0 0.6875;0 0 0.75;0 0 0.8125;0 0 0.875;0 0 0.9375;0 0 1;0 0.0625 1;0 0.125 1;0 0.1875 1;0 0.25 1;0 0.3125 1;0 0.375 1;0 0.4375 1;0 0.5 1;0 0.5625 1;0 0.625 1;0 0.6875 1;0 0.75 1;0 0.8125 1;0 0.875 1;0 0.9375 1;0 1 1;0.0625 1 1;0.125 1 0.9375;0.1875 1 0.875;0.25 1 0.8125;0.3125 1 0.75;0.375 1 0.6875;0.4375 1 0.625;0.5 1 0.5625;0.5625 1 0.5;0.625 1 0.4375;0.6875 1 0.375;0.75 1 0.3125;0.8125 1 0.25;0.875 1 0.1875;0.9375 1 0.125;1 1 0.0625;1 1 0;1 0.9375 0;1 0.875 0;1 0.8125 0;1 0.75 0;1 0.6875 0;1 0.625 0;1 0.5625 0;1 0.5 0;1 0.4375 0;1 0.375 0;1 0.3125 0;1 0.25 0;1 0.1875 0;1 0.125 0;1 0.0625 0;1 0 0;0.9375 0 0;0.875 0 0;0.8125 0 0;0.75 0 0;0.6875 0 0;0.625 0 0;0.5625 0 0],...
'HandleVisibility','on',...
'IntegerHandle','off',...
'InvertHardcopy',get(0,'defaultfigureInvertHardcopy'),...
'MenuBar','none',...
'Name','demo',...
'NumberTitle','off',...
'PaperPosition',get(0,'defaultfigurePaperPosition'),...
'Position',[0 0 GetWidth() GetHeight()],...
'Renderer',get(0,'defaultfigureRenderer'),...
'RendererMode','manual',...
'Resize','on',...
'Tag','figure1',...
'UserData',zeros(1,0));

setappdata(h1, 'GUIDEOptions', struct(...
'active_h', 1.270002e+002, ...
'taginfo', struct(...
'figure', 2, ...
'axes', 2, ...
'pushbutton', 6, ...
'popupmenu', 2, ...
'listbox', 4, ...
'text', 8, ...
'edit', 5), ...
'override', 0, ...
'release', 13, ...
'resize', 'none', ...
'accessibility', 'callback', ...
'mfile', 1, ...
'callbacks', 1, ...
'Position',[0.058 0.045 0.585 0.5],...
'TickDir',get(0,'defaultaxesTickDir'),...
'singleton', 1, ...
'syscolorfig', 1, ...
'lastSavedFile', 'C:\fastlib2\mlpack\mlpackdemo\demo.m'));


h2 = axes(...
'Parent',h1,...
'ALim',get(0,'defaultaxesALim'),...
'ALimMode','manual',...
'CameraPosition',[0.5 0.5 9.16025403784439],...
'CameraPositionMode','manual',...
'CameraTarget',[0.5 0.5 0.5],...
'CameraTargetMode','manual',...
'CameraUpVector',[0 1 0],...
'CameraUpVectorMode','manual',...
'CameraViewAngle',6.60861036031192,...
'CameraViewAngleMode','manual',...
'CLim',get(0,'defaultaxesCLim'),...
'CLimMode','manual',...
'Color',get(0,'defaultaxesColor'),...
'ColorOrder',get(0,'defaultaxesColorOrder'),...
'DataAspectRatio',get(0,'defaultaxesDataAspectRatio'),...
'DataAspectRatioMode','manual',...
'PlotBoxAspectRatio',get(0,'defaultaxesPlotBoxAspectRatio'),...
'PlotBoxAspectRatioMode','manual',...
'Position',[0.058 0.045 0.585 0.5],...
'TickDir',get(0,'defaultaxesTickDir'),...
'TickDirMode','manual',...
'XColor',get(0,'defaultaxesXColor'),...
'XLim',get(0,'defaultaxesXLim'),...
'XLimMode','manual',...
'XTick',[0 0.2 0.4 0.6 0.8 1],...
'XTickLabel',{ '0  ' '0.2' '0.4' '0.6' '0.8' '1  ' },...
'XTickLabelMode','manual',...
'XTickMode','manual',...
'YColor',get(0,'defaultaxesYColor'),...
'YLim',get(0,'defaultaxesYLim'),...
'YLimMode','manual',...
'YTick',[0 0.2 0.4 0.6 0.8 1],...
'YTickLabel',{ '0  ' '0.2' '0.4' '0.6' '0.8' '1  ' },...
'YTickLabelMode','manual',...
'YTickMode','manual',...
'ZColor',get(0,'defaultaxesZColor'),...
'ZLim',get(0,'defaultaxesZLim'),...
'ZLimMode','manual',...
'ZTick',[0 0.5 1],...
'ZTickLabel','',...
'ZTickLabelMode','manual',...
'ZTickMode','manual',...
'CreateFcn','demo(''axis_CreateFcn'',gcbo,[],guidata(gcbo))',...
'Tag','axes',...
'UserData',zeros(1,0));


h3 = get(h2,'title');

set(h3,...
'Parent',h2,...
'Color',[0 0 0],...
'HorizontalAlignment','center',...
'Position',[0.497175141242938 1.0183615819209 1.00005459937205],...
'VerticalAlignment','bottom',...
'HandleVisibility','off');

h4 = get(h2,'xlabel');

set(h4,...
'Parent',h2,...
'Color',[0 0 0],...
'HorizontalAlignment','center',...
'Position',[0.497175141242938 -0.0663841807909604 1.00005459937205],...
'VerticalAlignment','cap',...
'HandleVisibility','off');

h5 = get(h2,'ylabel');

set(h5,...
'Parent',h2,...
'Color',[0 0 0],...
'HorizontalAlignment','center',...
'Position',[-0.0819209039548023 0.495762711864407 1.00005459937205],...
'Rotation',90,...
'VerticalAlignment','bottom',...
'HandleVisibility','off');

h6 = get(h2,'zlabel');

set(h6,...
'Parent',h2,...
'Color',[0 0 0],...
'HorizontalAlignment','right',...
'Position',[-0.581920903954802 1.65395480225989 1.00005459937205],...
'HandleVisibility','off',...
'Visible','off');

h7 = uimenu(...
'Parent',h1,...
'Callback','demo(''FileMenu_Callback'',gcbo,[],guidata(gcbo))',...
'Label','File',...
'Tag','FileMenu');

h8 = uimenu(...
'Parent',h7,...
'Callback','demo(''OpenMenuItem_Callback'',gcbo,[],guidata(gcbo))',...
'Label','Open ...',...
'Tag','OpenMenuItem');

h9 = uimenu(...
'Parent',h7,...
'Callback','demo(''PrintMenuItem_Callback'',gcbo,[],guidata(gcbo))',...
'Label','Print ...',...
'Tag','PrintMenuItem');

h10 = uimenu(...
'Parent',h7,...
'Callback','demo(''CloseMenuItem_Callback'',gcbo,[],guidata(gcbo))',...
'Label','Close',...
'Separator','on',...
'Tag','CloseMenuItem');

h11 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'Callback','demo(''dim_reduction_Callback'',gcbo,[],guidata(gcbo))',...
'Position',[0.041 0.885 0.1 0.06],...
'String',GetReductionAlgorithms(),...
'Style','listbox',...
'Value',1,...
'CreateFcn','demo(''dim_reduction_CreateFcn'',gcbo,[],guidata(gcbo))',...
'Tag','dim_reduction1');


h12 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'Callback','demo(''neighbors_Callback'',gcbo,[],guidata(gcbo))',...
'Position',[0.165 0.885 0.108 0.06],...
'String',GetNeighborAlgorithms(),...
'Style','listbox',...
'Value',1,...
'CreateFcn','demo(''neighbors_CreateFcn'',gcbo,[],guidata(gcbo))',...
'Tag','neighbors1');


h13 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'Callback','demo(''algorithms_Callback'',gcbo,[],guidata(gcbo))',...
'Position',[0.355 0.885 0.1 0.06],...
'String',GetAlgorithms(),...
'Style','listbox',...
'Value',1,...
'CreateFcn','demo(''algorithms_CreateFcn'',gcbo,[],guidata(gcbo))',...
'Tag','algorithms1');


h14 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'ListboxTop',0,...
'Position',[0.049 0.799 0.1 0.037],...
'String','Data file',...
'Style','text',...
'Tag','data_file');


h15 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'BackgroundColor',[1 1 1],...
'Callback','demo(''data_file1_Callback'',gcbo,[],guidata(gcbo))',...
'Position',[0.049 0.7 0.497 0.11],...
'String',GetDataFiles(),...
'Style','listbox',...
'CreateFcn','demo(''data_file1_CreateFcn'',gcbo,[],guidata(gcbo))',...
'Tag','data_file1');


h16 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'ListboxTop',0,...
'Position',[0.041 0.944 0.1 0.022],...
'String','Dim Reduction',...
'Style','text',...
'Tag','dim_reduction');


h17 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'ListboxTop',0,...
'Position',[0.165 0.944 0.1 0.022],...
'String','Neighbors',...
'Style','text',...
'Tag','neighbors');


h18 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'ListboxTop',0,...
'Position',[0.355 0.944 0.1 0.022],...
'String','Algorithms',...
'Style','text',...
'Tag','algorithms');


h19 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'Callback','demo(''store_range_Callback'',gcbo,[],guidata(gcbo))',...
'ListboxTop',0,...
'Position',[0.466 0.82 0.072 0.025],...
'String','store_range',...
'Tag','store_range');


h20 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'Callback','demo(''go1_Callback'',gcbo,[],guidata(gcbo))',...
'ListboxTop',0,...
'Position',[0.041 0.849 0.065 0.027],...
'String','go',...
'Tag','go1');


h21 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'Callback','demo(''go2_Callback'',gcbo,[],guidata(gcbo))',...
'ListboxTop',0,...
'Position',[0.165 0.85 0.065 0.027],...
'String','go',...
'Tag','go2');


h22 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'Callback','demo(''go3_Callback'',gcbo,[],guidata(gcbo))',...
'ListboxTop',0,...
'Position',[0.356 0.849 0.065 0.027],...
'String','go',...
'Tag','go3');


h23 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'BackgroundColor',[1 1 1],...
'Callback','demo(''knn1_Callback'',gcbo,[],guidata(gcbo))',...
'ListboxTop',0,...
'Position',[0.289 0.905 0.041 0.04],...
'String','1',...
'Style','edit',...
'CreateFcn','demo(''knn1_CreateFcn'',gcbo,[],guidata(gcbo))',...
'Tag','knn1');


h24 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'ListboxTop',0,...
'Position',[0.289 0.944 0.05 0.017],...
'String','knn',...
'Style','text',...
'Tag','knn');


h25 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'BackgroundColor',[1 1 1],...
'Callback','demo(''bandwidth1_Callback'',gcbo,[],guidata(gcbo))',...
'ListboxTop',0,...
'Position',[0.289 0.842 0.054 0.048],...
'String','0.3',...
'Style','edit',...
'CreateFcn','demo(''bandwidth1_CreateFcn'',gcbo,[],guidata(gcbo))',...
'Tag','bandwidth1');


h26 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'ListboxTop',0,...
'Position',[0.289 0.889 0.058 0.017],...
'String','bandwidth',...
'Style','text',...
'Tag','bandwidth');

h27 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'ListboxTop',0,...
'Position',[0.467 0.944 0.058 0.017],...
'String','range',...
'Style','text',...
'Tag','range');

h28 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'BackgroundColor',[1 1 1],...
'Callback','demo(''range11_Callback'',gcbo,[],guidata(gcbo))',...
'ListboxTop',0,...
'Position',[0.467 0.9 0.054 0.048],...
'String','',...
'Style','edit',...
'CreateFcn','demo(''range11_CreateFcn'',gcbo,[],guidata(gcbo))',...
'Tag','range11');

h29 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'BackgroundColor',[1 1 1],...
'Callback','demo(''range12_Callback'',gcbo,[],guidata(gcbo))',...
'ListboxTop',0,...
'Position',[0.537 0.9 0.054 0.048],...
'String','',...
'Style','edit',...
'CreateFcn','demo(''range12_CreateFcn'',gcbo,[],guidata(gcbo))',...
'Tag','range12');

h30 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'BackgroundColor',[1 1 1],...
'Callback','demo(''range21_Callback'',gcbo,[],guidata(gcbo))',...
'ListboxTop',0,...
'Position',[0.467 0.85 0.054 0.048],...
'String','',...
'Style','edit',...
'CreateFcn','demo(''range21_CreateFcn'',gcbo,[],guidata(gcbo))',...
'Tag','range21');

h31 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'BackgroundColor',[1 1 1],...
'Callback','demo(''range22_Callback'',gcbo,[],guidata(gcbo))',...
'ListboxTop',0,...
'Position',[0.537 0.85 0.054 0.048],...
'String','',...
'Style','edit',...
'CreateFcn','demo(''range22_CreateFcn'',gcbo,[],guidata(gcbo))',...
'Tag','range22');

h32 = axes(...
'Parent',h1,...
'ALim',get(0,'defaultaxesALim'),...
'ALimMode','manual',...
'CameraPosition',[0.5 0.5 9.16025403784439],...
'CameraPositionMode','manual',...
'CameraTarget',[0.5 0.5 0.5],...
'CameraTargetMode','manual',...
'CameraUpVector',[0 1 0],...
'CameraUpVectorMode','manual',...
'CameraViewAngle',6.60861036031192,...
'CameraViewAngleMode','manual',...
'CLim',get(0,'defaultaxesCLim'),...
'CLimMode','manual',...
'Color',get(0,'defaultaxesColor'),...
'ColorOrder',get(0,'defaultaxesColorOrder'),...
'DataAspectRatio',get(0,'defaultaxesDataAspectRatio'),...
'DataAspectRatioMode','manual',...
'PlotBoxAspectRatio',get(0,'defaultaxesPlotBoxAspectRatio'),...
'PlotBoxAspectRatioMode','manual',...
'Position',[0.65 0.6 0.3 0.3],...
'TickDir',get(0,'defaultaxesTickDir'),...
'TickDirMode','manual',...
'XColor',get(0,'defaultaxesXColor'),...
'XLim',get(0,'defaultaxesXLim'),...
'XLimMode','manual',...
'XTick',[0 0.2 0.4 0.6 0.8 1],...
'XTickLabel',{ '0  ' '0.2' '0.4' '0.6' '0.8' '1  ' },...
'XTickLabelMode','manual',...
'XTickMode','manual',...
'YColor',get(0,'defaultaxesYColor'),...
'YLim',get(0,'defaultaxesYLim'),...
'YLimMode','manual',...
'YTick',[0 0.2 0.4 0.6 0.8 1],...
'YTickLabel',{ '0  ' '0.2' '0.4' '0.6' '0.8' '1  ' },...
'YTickLabelMode','manual',...
'YTickMode','manual',...
'ZColor',get(0,'defaultaxesZColor'),...
'ZLim',get(0,'defaultaxesZLim'),...
'ZLimMode','manual',...
'ZTick',[0 0.5 1],...
'ZTickLabel','',...
'ZTickLabelMode','manual',...
'ZTickMode','manual',...
'Tag','axes1',...
'CreateFcn','demo(''axis1_CreateFcn'',gcbo,[],guidata(gcbo))',...
'UserData',zeros(1,0));

h33 = uicontrol(...
'FontSize',32,...
'Parent',h1,...
'Units','normalized',...
'ListboxTop',0,...
'Position',[0.05 0.58 0.3 0.08],...
'String','Method running',...
'Style','text',...
'Tag','flashlight');

h34 = uicontrol(...
'Parent',h1,...
'Units','normalized',...
'Callback','demo(''textbox_output_Callback'',gcbo,[],guidata(gcbo))',...
'Position',[0.663 0.054 0.313 0.483],...
'String','Text output',...
'Style','listbox',...
'Value',1,...
'CreateFcn','demo(''textbox_output_CreateFcn'',gcbo,[],guidata(gcbo))',...
'Tag','textbox_output');


hsingleton = h1;


% --- Handles default GUIDE GUI creation and callback dispatch
function varargout = gui_mainfcn(gui_State, varargin)


%   GUI_MAINFCN provides these command line APIs for dealing with GUIs
%
%      DEMO_EXPORT, by itself, creates a new DEMO_EXPORT or raises the existing
%      singleton*.
%
%      H = DEMO_EXPORT returns the handle to a new DEMO_EXPORT or the handle to
%      the existing singleton*.
%
%      DEMO_EXPORT('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DEMO_EXPORT.M with the given input arguments.
%
%      DEMO_EXPORT('Property','Value',...) creates a new DEMO_EXPORT or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before untitled_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to untitled_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".

%   Copyright 1984-2002 The MathWorks, Inc.
%   $Revision: 1.4 $ $Date: 2002/05/31 21:44:31 $

gui_StateFields =  {'gui_Name'
                    'gui_Singleton'
                    'gui_OpeningFcn'
                    'gui_OutputFcn'
                    'gui_LayoutFcn'
                    'gui_Callback'};
gui_Mfile = '';
for i=1:length(gui_StateFields)
    if ~isfield(gui_State, gui_StateFields{i})
        error('Could not find field %s in the gui_State struct in GUI M-file %s', gui_StateFields{i}, gui_Mfile);        
    elseif isequal(gui_StateFields{i}, 'gui_Name')
        gui_Mfile = [getfield(gui_State, gui_StateFields{i}), '.m'];
    end
end

numargin = length(varargin);

if numargin == 0
    % DEMO_EXPORT
    % create the GUI
    gui_Create = 1;
elseif numargin > 3 & ischar(varargin{1}) & ishandle(varargin{2})
    % DEMO_EXPORT('CALLBACK',hObject,eventData,handles,...)
    gui_Create = 0;
else
    % DEMO_EXPORT(...)
    % create the GUI and hand varargin to the openingfcn
    gui_Create = 1;
end

if gui_Create == 0
    varargin{1} = gui_State.gui_Callback;
    if nargout
        [varargout{1:nargout}] = feval(varargin{:});
    else
        feval(varargin{:});
    end
else
    if gui_State.gui_Singleton
        gui_SingletonOpt = 'reuse';
    else
        gui_SingletonOpt = 'new';
    end
    
    % Open fig file with stored settings.  Note: This executes all component
    % specific CreateFunctions with an empty HANDLES structure.
    
    % Do feval on layout code in m-file if it exists
    if ~isempty(gui_State.gui_LayoutFcn)
        gui_hFigure = feval(gui_State.gui_LayoutFcn, gui_SingletonOpt);
    else
        gui_hFigure = local_openfig(gui_State.gui_Name, gui_SingletonOpt);            
        % If the figure has InGUIInitialization it was not completely created
        % on the last pass.  Delete this handle and try again.
        if isappdata(gui_hFigure, 'InGUIInitialization')
            delete(gui_hFigure);
            gui_hFigure = local_openfig(gui_State.gui_Name, gui_SingletonOpt);            
        end
    end
    
    % Set flag to indicate starting GUI initialization
    setappdata(gui_hFigure,'InGUIInitialization',1);

    % Fetch GUIDE Application options
    gui_Options = getappdata(gui_hFigure,'GUIDEOptions');
    
    if ~isappdata(gui_hFigure,'GUIOnScreen')
        % Adjust background color
        if gui_Options.syscolorfig 
            set(gui_hFigure,'Color', get(0,'DefaultUicontrolBackgroundColor'));
        end

        % Generate HANDLES structure and store with GUIDATA
        guidata(gui_hFigure, guihandles(gui_hFigure));
    end
    
    % If user specified 'Visible','off' in p/v pairs, don't make the figure
    % visible.
    gui_MakeVisible = 1;
    for ind=1:2:length(varargin)
        if length(varargin) == ind
            break;
        end
        len1 = min(length('visible'),length(varargin{ind}));
        len2 = min(length('off'),length(varargin{ind+1}));
        if ischar(varargin{ind}) & ischar(varargin{ind+1}) & ...
                strncmpi(varargin{ind},'visible',len1) & len2 > 1
            if strncmpi(varargin{ind+1},'off',len2)
                gui_MakeVisible = 0;
            elseif strncmpi(varargin{ind+1},'on',len2)
                gui_MakeVisible = 1;
            end
        end
    end
    
    % Check for figure param value pairs
    for index=1:2:length(varargin)
        if length(varargin) == index
            break;
        end
        try, set(gui_hFigure, varargin{index}, varargin{index+1}), catch, break, end
    end

    % If handle visibility is set to 'callback', turn it on until finished
    % with OpeningFcn
    gui_HandleVisibility = get(gui_hFigure,'HandleVisibility');
    if strcmp(gui_HandleVisibility, 'callback')
        set(gui_hFigure,'HandleVisibility', 'on');
    end
    
    feval(gui_State.gui_OpeningFcn, gui_hFigure, [], guidata(gui_hFigure), varargin{:});
    
    if ishandle(gui_hFigure)
        % Update handle visibility
        set(gui_hFigure,'HandleVisibility', gui_HandleVisibility);
        
        % Make figure visible
        if gui_MakeVisible
            set(gui_hFigure, 'Visible', 'on')
            if gui_Options.singleton 
                setappdata(gui_hFigure,'GUIOnScreen', 1);
            end
        end

        % Done with GUI initialization
        rmappdata(gui_hFigure,'InGUIInitialization');
    end
    
    % If handle visibility is set to 'callback', turn it on until finished with
    % OutputFcn
    if ishandle(gui_hFigure)
        gui_HandleVisibility = get(gui_hFigure,'HandleVisibility');
        if strcmp(gui_HandleVisibility, 'callback')
            set(gui_hFigure,'HandleVisibility', 'on');
        end
        gui_Handles = guidata(gui_hFigure);
    else
        gui_Handles = [];
    end
    
    if nargout
        [varargout{1:nargout}] = feval(gui_State.gui_OutputFcn, gui_hFigure, [], gui_Handles);
    else
        feval(gui_State.gui_OutputFcn, gui_hFigure, [], gui_Handles);
    end
    
    if ishandle(gui_hFigure)
        set(gui_hFigure,'HandleVisibility', gui_HandleVisibility);
    end
end    

function gui_hFigure = local_openfig(name, singleton)
if nargin('openfig') == 3 
    gui_hFigure = openfig(name, singleton, 'auto');
else
    % OPENFIG did not accept 3rd input argument until R13,
    % toggle default figure visible to prevent the figure
    % from showing up too soon.
    gui_OldDefaultVisible = get(0,'defaultFigureVisible');
    set(0,'defaultFigureVisible','off');
    gui_hFigure = openfig(name, singleton);
    set(0,'defaultFigureVisible',gui_OldDefaultVisible);
end

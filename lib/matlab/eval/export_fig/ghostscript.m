function varargout = ghostscript(cmd)
%GHOSTSCRIPT  Calls a local GhostScript executable with the input command
%
% Example:
%   [status result] = ghostscript(cmd)
%
% Attempts to locate a ghostscript executable, finally asking the user to
% specify the directory ghostcript was installed into. The resulting path
% is stored for future reference.
% 
% Once found, the executable is called with the input command string.
%
% This function requires that you have Ghostscript installed on your
% system. You can download this from: http://www.ghostscript.com
%
% IN:
%   cmd - Command string to be passed into ghostscript.
%
% OUT:
%   status - 0 iff command ran without problem.
%   result - Output from ghostscript.

% Copyright: Oliver Woodford, 2009-2010

% Thanks to Jonas Dorn for the fix for the title of the uigetdir window on
% Mac OS.

% Thanks to Nathan Childress for the fix to the default location on 64-bit
% Windows systems.

% 27/4/11 - Find 64-bit Ghostscript on Windows. Thanks to Paul Durack and
% Shaun Kline for pointing out the issue

% 4/5/11 - Thanks to David Chorlian for pointing out an alternative
% location for gs on linux.

% Call ghostscript
[varargout{1:nargout}] = system(sprintf('"%s" %s', gs_path, cmd));
return

function path = gs_path
% Return a valid path
% Start with the currently set path
path = user_string('ghostscript');
% Check the path works
if check_gs_path(path)
    return
end
% Check whether the binary is on the path
if ispc
    bin = {'gswin32c.exe', 'gswin64c.exe'};
else
    bin = {'gs'};
end
for a = 1:numel(bin)
    path = bin{a};
    if check_store_gs_path(path)
        return
    end
end
% Search the obvious places
if ispc
    default_location = 'C:\Program Files\gs\';
    dir_list = dir(default_location);
    if isempty(dir_list)
        default_location = 'C:\Program Files (x86)\gs\'; % Possible location on 64-bit systems 
        dir_list = dir(default_location);
    end
    executable = {'\bin\gswin32c.exe', '\bin\gswin64c.exe'};
    ver_num = 0;
    % If there are multiple versions, use the newest
    for a = 1:numel(dir_list)
        ver_num2 = sscanf(dir_list(a).name, 'gs%g');
        if ~isempty(ver_num2) && ver_num2 > ver_num
            for b = 1:numel(executable)
                path2 = [default_location dir_list(a).name executable{b}];
                if exist(path2, 'file') == 2
                    path = path2;
                    ver_num = ver_num2;
                end
            end
        end
    end
    if check_store_gs_path(path)
        return
    end
else
    bin = {'/usr/bin/gs', '/usr/local/bin/gs'};
    for a = 1:numel(bin)
        path = bin{a};
        if check_store_gs_path(path)
            return
        end
    end
end
% Ask the user to enter the path
while 1
    if strncmp(computer, 'MAC', 3) % Is a Mac
        % Give separate warning as the uigetdir dialogue box doesn't have a
        % title
        uiwait(warndlg('Ghostscript not found. Please locate the program.'))
    end
    base = uigetdir('/', 'Ghostcript not found. Please locate the program.');
    if isequal(base, 0)
        % User hit cancel or closed window
        break;
    end
    base = [base filesep];
    bin_dir = {'', ['bin' filesep], ['lib' filesep]};
    for a = 1:numel(bin_dir)
        for b = 1:numel(bin)
            path = [base bin_dir{a} bin{b}];
            if exist(path, 'file') == 2
                if check_store_gs_path(path)
                    return
                end
            end
        end
    end
end
error('Ghostscript not found. Have you installed it from www.ghostscript.com?');

function good = check_store_gs_path(path)
% Check the path is valid
good = check_gs_path(path);
if ~good
    return
end
% Update the current default path to the path found
if ~user_string('ghostscript', path)
    warning('Path to ghostscript installation could not be saved. Enter it manually in ghostscript.txt.');
    return
end
return

function good = check_gs_path(path)
% Check the path is valid
[good message] = system(sprintf('"%s" -h', path));
good = good == 0;
return
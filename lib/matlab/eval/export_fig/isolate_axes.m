%ISOLATE_AXES Isolate the specified axes in a figure on their own
%
% Examples:
%   fh = isolate_axes(ah)
%   fh = isolate_axes(ah, vis)
%
% This function will create a new figure containing the axes specified, and
% also their associated legends and colorbars. The axes specified must all
% be in the same figure, but they will generally only be a subset of the
% axes in the figure.
%
% IN:
%    ah - An array of axes handles, which must come from the same figure.
%    vis - A boolean indicating whether the new figure should be visible.
%          Default: false.
%
% OUT:
%    fh - The handle of the created figure.

% Copyright (C) Oliver Woodford 2011-2012

% Thank you to Rosella Blatt for reporting a bug to do with axes in GUIs
% 16/3/2012 Moved copyfig to its own function. Thanks to Bob Fratantonio
% for pointing out that the function is also used in export_fig.m.

function fh = isolate_axes(ah, vis)
% Make sure we have an array of handles
if ~all(ishandle(ah))
    error('ah must be an array of handles');
end
% Check that the handles are all for axes, and are all in the same figure
fh = ancestor(ah(1), 'figure');
nAx = numel(ah);
for a = 1:nAx
    if ~strcmp(get(ah(a), 'Type'), 'axes')
        error('All handles must be axes handles.');
    end
    if ~isequal(ancestor(ah(a), 'figure'), fh)
        error('Axes must all come from the same figure.');
    end
end
% Tag the axes so we can find them in the copy
old_tag = get(ah, 'Tag');
if nAx == 1
    old_tag = {old_tag};
end
set(ah, 'Tag', 'ObjectToCopy');
% Create a new figure exactly the same as the old one
fh = copyfig(fh); %copyobj(fh, 0);
if nargin < 2 || ~vis
    set(fh, 'Visible', 'off');
end
% Reset the axes tags
for a = 1:nAx
    set(ah(a), 'Tag', old_tag{a});
end
% Find the objects to save
ah = findall(fh, 'Tag', 'ObjectToCopy');
if numel(ah) ~= nAx
    close(fh);
    error('Incorrect number of axes found.');
end
% Set the axes tags to what they should be
for a = 1:nAx
    set(ah(a), 'Tag', old_tag{a});
end
% Keep any legends and colorbars which overlap the subplots
lh = findall(fh, 'Type', 'axes', '-and', {'Tag', 'legend', '-or', 'Tag', 'Colorbar'});
nLeg = numel(lh);
if nLeg > 0
    ax_pos = get(ah, 'OuterPosition');
    if nAx > 1
        ax_pos = cell2mat(ax_pos(:));
    end
    ax_pos(:,3:4) = ax_pos(:,3:4) + ax_pos(:,1:2);
    leg_pos = get(lh, 'OuterPosition');
    if nLeg > 1;
        leg_pos = cell2mat(leg_pos);
    end
    leg_pos(:,3:4) = leg_pos(:,3:4) + leg_pos(:,1:2);
    for a = 1:nAx
            % Overlap test
            ah = [ah; lh(leg_pos(:,1) < ax_pos(a,3) & leg_pos(:,2) < ax_pos(a,4) &...
                         leg_pos(:,3) > ax_pos(a,1) & leg_pos(:,4) > ax_pos(a,2))];
    end
end
% Get all the objects in the figure
axs = findall(fh);
% Delete everything except for the input axes and associated items
delete(axs(~ismember(axs, [ah; allchildren(ah); allancestors(ah)])));
return

function ah = allchildren(ah)
ah = allchild(ah);
if iscell(ah)
    ah = cell2mat(ah);
end
ah = ah(:);
return

function ph = allancestors(ah)
ph = [];
for a = 1:numel(ah)
    h = get(ah(a), 'parent');
    while h ~= 0
        ph = [ph; h];
        h = get(h, 'parent');
    end
end
return
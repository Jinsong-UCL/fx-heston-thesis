function root=setupHeston()
% Add only this codebase's numerical routines to the MATLAB path.
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'matlab','common'));
addpath(fullfile(root,'matlab','matrix'));
addpath(fullfile(root,'matlab','vector'));
addpath(fullfile(root,'examples'));
addpath(fullfile(root,'tests'));
end

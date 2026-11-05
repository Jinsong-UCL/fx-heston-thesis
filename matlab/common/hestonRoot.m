function root=hestonRoot()
% Resolve the repository root independently of MATLAB's working directory.
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
end

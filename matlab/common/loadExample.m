function data=loadExample(name)
% Load a compact January 2013 numerical example.
assert(any(strcmp(name,{'matrix','vector_N1','vector_N2'})),'Unknown example.');
data=jsondecode(fileread(fullfile(hestonRoot(),'data',[name '.json'])));
end

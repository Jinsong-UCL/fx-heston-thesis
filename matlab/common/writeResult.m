function writeResult(path,data)
% Save a numerical result; create its output directory if necessary.
folder=fileparts(path);if ~isempty(folder) && ~exist(folder,'dir'),mkdir(folder);end
fid=fopen(path,'w');assert(fid>=0,'Cannot open output file: %s',path);
cleanup=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(data,'PrettyPrint',true));
end

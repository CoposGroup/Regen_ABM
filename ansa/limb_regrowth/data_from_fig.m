fig = openfig('data/input/exp_CTRL/T12_data/C59_T12.fig');

% Get all line objects
lines = findobj(fig, 'Type', 'line');

outDir = 'curve_csvs';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

for k = 1:length(lines)
    x = lines(k).XData(:);
    y = lines(k).YData(:);
    data = [x y];

    % Get the label (DisplayName) if it exists
    name = lines(k).DisplayName;
    
    if isempty(name)
        % Fallback if no label was set
        name = sprintf('curve_%d', k);
    end
    
    % Sanitize filename (remove illegal characters for file names)
    name = regexprep(name, '[^a-zA-Z0-9_-]', '_');

    outFile = fullfile(outDir, [name '.csv']);
    
    % Write with headers
    fid = fopen(outFile, 'w');
    fprintf(fid, 'x,y\n'); % header
    fclose(fid);
    dlmwrite(outFile, data, '-append');
    
    fprintf('Saved curve %d as %s\n', k, outFile);
end

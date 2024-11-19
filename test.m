close all;

% openfig('Hypotheses/Morphogen/N=500/morphogen.fig');
% 
% h = gcf;
% 
% lines = findobj(h, 'Type', 'Line');
% 
% for i = 1:length(lines)
%     get(lines(i), 'Color')
% end
% 
% for i = 1:length(lines)
%     if ~isequal(get(lines(i), 'Color'), [0 0 0])
%         delete(lines(i));
%     end
% end
% 
% 
% savefig(h, 'black_line_only.fig');





fig = openfig('Hypotheses/Morphogen/N=500/morphogen.fig');

axObjs = fig.Children;
dataObjs = axObjs.Children;

n = dataObjs(end - 1);



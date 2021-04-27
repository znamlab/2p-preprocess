function rois = correct_neuropil(dpath, ops)
% INPUTS: path to suite2p preprocessed imaging data ('Fall.mat' and 'neu_masks.mat')
%     and ops struct containing flag for fast neuropil
% OUTPUT: function saves roi structure containing:
%     footprint -
%     footprint_neuropil -
%     activity -
%     neuropil -
%     activity_trend -detrended with [10 percentile, 2000ms half] window, median subtracted
%     neuropil_trend -
%     cleaned -
%     dff -
%     f0 -
%     spikes -
%     iscell -
% called rois.mat when fast neuropil is used or rois_astm_neuropil.mat when
% neuropil is removed using AST model.

% Add paths
codepath = fileparts(mfilename('fullpath'));
addpath(fullfile(codepath, '../thirdparty/ast_model/src/'))
addpath(fullfile(codepath, '../thirdparty/runperc/'));
addpath(fullfile(codepath, 'utils'))

% Set options
defaults.cells_only = false;
defaults.fast_neuropil = false;
defaults.trend_halfwin = 2000;
defaults.trend_perc = 10;
defaults.gmm_comps = 3;

if nargin>=2 && ~isempty(ops)
    ops = parse_ops(ops, defaults);
else
    ops = defaults;
end

% Load data
dfile = 'Fall.mat';
dvar = {'F','iscell','ops','Fneu','spks','stat'};
data = load(fullfile(dpath, dfile), dvar{:});
% TODO get the neuropil masks code
%masks = load(fullfile(dpath, 'neu_masks.mat'));

% Get neuropil masks from suite2p data
Lx = data.ops(1).Lx;
Ly = data.ops(1).Ly;

stat = data.stat;
F = data.F;
Fneu = data.Fneu;
spks = data.spks;
iscell = data.iscell;

ncells = size(F,1);
% neuropil_pix = sum(sum(masks.neuropil_masks>0, 3), 2);
% for indR = 1:size(masks.neuropil_masks, 1)
%     neuropil_masks{indR} = sparse(double(squeeze(masks.neuropil_masks(indR,:,:))))'>0;
% end

% merged_cells = [];
% for indR = 1:ncells
%     if isfield(stat{indR}, 'imerge') && ~isempty(stat{indR}.imerge)
%         in_merge = double(stat{indR}.imerge+1);
%         merged_cells = [ merged_cells in_merge ];
%         % offset for 1-based indexing
%         neuropil_pix(indR) = mean(neuropil_pix(in_merge));
%         neuropil_masks{indR} = zeros(Lx, Ly);
%         for indM = in_merge
%             neuropil_masks{indR} = neuropil_masks{indR} + ...
%                 neuropil_masks{indM};
%         end
%         neuropil_masks{indR} = neuropil_masks{indR} / numel(in_merge);
%     end
% end

if isempty(gcp('nocreate'))
    parpool(8);
end

rois = repmat(struct(...
    'footprint', [], ...
    'footprint_neuropil', [], ...
    'activity', [], ...
    'neuropil', [], ...
    'activity_trend', [], ...
    'neuropil_trend', [], ...
    'cleaned', [], ...
    'dff', [], ...
    'f0', [], ...
    'spikes', [], ...
    'iscell', []), ncells, 1);

% Loop trough ROIs
parfor indR = 1:ncells
    if iscell(indR) || ~ops.cells_only % proceed for only cell tagged ROIs
        tic
        rois(indR).footprint = make_footprint(stat{indR}.xpix, stat{indR}.ypix, Lx, Ly);
        rois(indR).activity = F(indR,:);
        rois(indR).neuropil = Fneu(indR,:);
        % rois(indR).footprint_neuropil = neuropil_masks{indR};

        Fpix = numel(stat{indR}.xpix); % cell footprint in pixels

        rois(indR).activity_trend = running_percentile(rois(indR).activity, ...
            ops.trend_halfwin * 2, ops.trend_perc)';
        rois(indR).activity_trend = rois(indR).activity_trend - ...
            median(rois(indR).activity_trend);
        rois(indR).neuropil_trend = running_percentile(rois(indR).neuropil, ...
            ops.trend_halfwin * 2, ops.trend_perc)';
        rois(indR).neuropil_trend = rois(indR).neuropil_trend - ...
            median(rois(indR).neuropil_trend);
        if ops.fast_neuropil
            rois(indR).cleaned = (rois(indR).activity - rois(indR).activity_trend) - ...
                0.7 * (rois(indR).neuropil - rois(indR).neuropil_trend);
        else
            n_sectors = round([Fpix 100]); %neuropil_pix(indR)
            rois(indR).cleaned = fit_ast_model(...
                [rois(indR).activity - rois(indR).activity_trend;...
                rois(indR).neuropil - rois(indR).neuropil_trend], ...
                n_sectors, 'detrend', 'none');
        end
        [rois(indR).dff, rois(indR).f0] = ...
            extractdff_gmm(rois(indR).cleaned, 'ncomps', 3);
        rois(indR).spikes = spks(indR, :);
        rois(indR).iscell = iscell(indR);

        t = toc;
        fprintf('Extracted cell %d of %d in %f s...\n', ...
            indR, ncells, t);
    end
end

%rois(merged_cells) = [];

if ops.fast_neuropil
    savefile = fullfile(dpath, 'rois.mat');
else
    savefile = fullfile(dpath, 'rois_astm_neuropil.mat');
end

fprintf('Saving to %s...\n', savefile);
save(savefile, 'rois', 'ops', '-v7.3');
fprintf('All done!\n');

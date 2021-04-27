function [dff, f0] = extractdff_gmm(traces, varargin)
    % EXTRACTDFF_GMM transform raw traces into dF/F0 traces using GMM
    %
    % [dff, f0] = extractdff_prc(traces)
    %
    % This function uses a Gaussian mixture model (GMM) to estimate the F0
    % baseline.
    %
    % INPUTS
    %   traces - raw signals, as a [#ROIs Time] array
    %
    % NAME-VALUE PAIR INPUTS (optional)
    %   maxiter - default: 1000
    %       maximum number of iterations to fit the GMM
    %   ncomps - default: 2
    %       number of mixture components
    %   seed - default: 12345
    %       random number generator seed, fixed for reproducibility
    %
    % OUTPUTS
    %   dff - delta F over F0, as a [#ROIs Time] array
    %   f0 - F0 baseline, as a [#ROIs 1] array
    %
    % EXAMPLES
    %   % extract dF/F0 for each ROI
    %   [dff, f0] = roisfilter(rois, @extractdff_gmm);
    %
    % SEE ALSO roisfilter, stacksextract

    % Author: Maxime Rio

    if ~exist('traces', 'var')
        error('Missing traces argument.')
    end
    validateattributes(traces, {'numeric'}, {'nonempty', '2d'}, '', 'traces');
    [nrois, ~] = size(traces);

    % parse optional inputs
    parser = inputParser;
    posint_attr = {'scalar', 'integer', 'positive'};
    parser.addParameter('maxiter', 2000, ...
        @(x) validateattributes(x, {'numeric'}, posint_attr, '', 'maxiter'));
    parser.addParameter('ncomps', 2, ...
        @(x) validateattributes(x, {'numeric'}, posint_attr, '', 'ncomps'));
    parser.addParameter('seed', 12345, ...
        @(x) validateattributes(x, {'numeric'}, posint_attr, '', 'seed'));

    parser.parse(varargin{:});
    maxiter = parser.Results.maxiter;
    ncomps = parser.Results.ncomps;
    seed = parser.Results.seed;

    % fix the RNG seed for reproducibility
    rng(seed);

    % for a each trace, fit a GMM and keep the lowest mean
    f0 = nan(nrois, 1);
    for ii = 1:nrois
        % skip NaN traces (representing empty ROIs or too close to borders)
        if sum(~isnan(traces(ii, :))) < ncomps + 1
            continue;
        end
        reg_value = nanvar(traces(ii, :)) * 1e-10;  % avoid unstable fit
        gmm_opts = statset('MaxIter', maxiter);
        obj = fitgmdist(traces(ii, :)', ncomps, ...
            'Options', gmm_opts, 'RegularizationValue', reg_value);
        f0(ii) = min(obj.mu);
    end

    dff = (traces - f0) ./ f0;
end

function ops_out = parse_ops(ops, defaults)
% INPUTS:
%     ops - options structure to override defaults
%     defaults - structure with default options
% OUTPUTS:
%     ops_out - structure containing updated options
fields = fieldnames(defaults);

for indF = 1:numel(fields)
    if isfield(ops, fields{indF})
        ops_out.(fields{indF}) = ops.(fields{indF});
    else
        ops_out.(fields{indF}) = defaults.(fields{indF});
    end
end

function manifest = export_correlator_records(configFile)
%EXPORT_CORRELATOR_RECORDS Export raw I-branch epochs for Python experiments.
% Run from repository root after editing a configs/data/*.example.json copy.
% No normalization or random split is performed here. Python partitions raw
% state intervals before constructing 10 ms non-overlapping windows.
%
% processed_start_original_s is the original-signal time of the FIRST
% SELECTED epoch. read_start_epoch (optional, default 1) selects that epoch
% within trackData. These are explicit; no undocumented initial skip occurs.
% Mixed-record event times are relative to the first tracking epoch instead.
% Adapted from data_generate_mixed_order_transfer_20261020.m.

    cfg = jsondecode(fileread(configFile));
    validateattributes(cfg.correlators, {'numeric'}, ...
        {'scalar','integer','>=',3,'<=',81});
    assert(mod(cfg.correlators,2)==1,'Use an odd number of correlators.');
    assert(numel(unique(string({cfg.records.id})))==numel(cfg.records), ...
        'Record IDs must be unique.');
    for k=1:numel(cfg.records)
        r=cfg.records(k);
        assert(~isempty(r.mat_file)&&isfile(r.mat_file), ...
            'Set an existing mat_file for %s.',r.id);
        validateattributes(r.channel,{'numeric'},{'scalar','integer','positive'});
        validateattributes(r.duration_s,{'numeric'},{'scalar','positive','finite'});
        validateattributes(r.processed_start_original_s,{'numeric'}, ...
            {'scalar','nonnegative','finite'});
        assert(abs(r.duration_s*1000-round(r.duration_s*1000))<1e-6, ...
            'duration_s must be aligned to 1 ms.');
        assert(~isfile(fullfile(cfg.output_dir,[r.id '.h5'])), ...
            'Output exists; choose a new output_dir.');
    end
    if ~isfolder(cfg.output_dir), mkdir(cfg.output_dir); end
    rows=cell(numel(cfg.records),1);
    for k=1:numel(cfg.records)
        rows{k}=exportOne(cfg,cfg.records(k));
    end
    manifest=vertcat(rows{:});
    writetable(manifest,fullfile(cfg.output_dir,'dataset_manifest.csv'));
end

function row=exportOne(cfg,r)
    count=round(r.duration_s*1000);
    first=1;
    if isfield(r,'read_start_epoch')&&~isempty(r.read_start_epoch)
        first=r.read_start_epoch;
    end
    validateattributes(first,{'numeric'},{'scalar','integer','positive'});
    base=['/trackData/' r.system '/channel'];
    % Preserve the original selective v7.3 reader, with a v7 fallback.
    isHdf=true;
    try
        info=h5info(r.mat_file,base);
        fields={info.Datasets.Name};
    catch
        isHdf=false;
        loaded=load(r.mat_file,'trackData');
        chan=loaded.trackData.(r.system).channel(r.channel);
        fields=fieldnames(chan);
    end
    [names,indices,offsets]=selectFields(fields,cfg.correlators);
    values=zeros(cfg.correlators,count,'single');
    for j=1:cfg.correlators
        if isHdf
            path=referencedPath(r.mat_file,[base '/' names{j}],r.channel);
            vector=readVectorRange(r.mat_file,path,first,count);
        else
            vector=chan.(names{j});
            assert(numel(vector)>=first+count-1,'Recording is shorter than requested.');
            vector=vector(first:first+count-1);
        end
        values(j,:)=single(vector(:).');
    end
    assert(all(isfinite(values(:))),'Nonfinite correlator output.');
    % Event boundaries are half-open: [start,end), with post-entry labels at equality.
    time=r.processed_start_original_s+(0:count-1)'/1000;
    mp=false(count,1); sp=false(count,1);
    if ~isempty(r.multipath_entry_original_s)
        mp=time>=r.multipath_entry_original_s;
    end
    if ~isempty(r.spoofing_entry_original_s)
        sp=time>=r.spoofing_entry_original_s;
    end
    state=int8(sp)+2*int8(mp);
    [~,file,extension]=fileparts(r.mat_file);
    meta=struct('SchemaVersion',4,'RecordID',r.id,'SourceFile',[file extension], ...
        'Channel',r.channel,'System',r.system,'Correlators',cfg.correlators, ...
        'SelectedIFields',{names},'SelectedCorrelatorIndices',indices, ...
        'SelectedCorrelatorOffsets_chips',offsets,'EpochInterval_ms',1, ...
        'AmplitudeNormalization','none','ReadStartEpoch',first, ...
        'ProcessedStartOriginal_s',r.processed_start_original_s, ...
        'TimeReference',r.time_reference,'TotalTime_ms',count, ...
        'MultipathEntryOriginal_s',r.multipath_entry_original_s, ...
        'SpoofingEntryOriginal_s',r.spoofing_entry_original_s, ...
        'FeatureRepresentation','raw signed I; Python applies abs(I)', ...
        'DataLayout','MATLAB [correlators,epochs]; h5py [epochs,correlators]');
    destination=fullfile(cfg.output_dir,[r.id '.h5']);
    h5create(destination,'/I',size(values),'Datatype','single', ...
        'ChunkSize',[cfg.correlators,min(count,1000)],'Deflate',5);
    h5write(destination,'/I',values);
    h5create(destination,'/physical_state',[count,1],'Datatype','int8');
    h5write(destination,'/physical_state',state);
    h5writeatt(destination,'/','Metadata',jsonencode(meta));
    assert(isequal(h5read(destination,'/I'),values),'Readback validation failed.');
    row=table(string(r.id),r.channel,count,r.processed_start_original_s, ...
        sum(state==0),sum(state==1),sum(state==2),sum(state==3), ...
        'VariableNames',{'record','channel','epochs','original_start_s','A','AS','AM','AMS'});
    fprintf('Exported %s: %d epochs, %d correlators.\n',r.id,count,cfg.correlators);
end

function [names,indices,offsets]=selectFields(fields,n)
    % FGI-GSRx convention for 81 taps: E_39,...,E,P,L,...,L_39.
    assert(all(ismember({'I_E_39','I_E','I_P','I_L','I_L_39'},fields)), ...
        'Expected the documented 81-correlator FGI-GSRx interface.');
    indices=round(linspace(1,81,n));
    assert(numel(unique(indices))==n,'Duplicate correlator selections.');
    grid=linspace(-2,2,81); offsets=grid(indices); names=cell(1,n);
    for j=1:n
        ix=indices(j);
        if ix<41
            k=40-ix; suffix='E';
            if k>0,suffix=sprintf('E_%d',k);end
        elseif ix==41
            suffix='P';
        else
            k=ix-42; suffix='L';
            if k>0,suffix=sprintf('L_%d',k);end
        end
        names{j}=['I_' suffix];
        assert(ismember(names{j},fields),'Missing correlator field %s.',names{j});
    end
end

function path=referencedPath(file,dataset,channel)
    fid=H5F.open(file,'H5F_ACC_RDONLY','H5P_DEFAULT');
    closeFile=onCleanup(@() H5F.close(fid)); %#ok<NASGU>
    did=H5D.open(fid,dataset);
    closeData=onCleanup(@() H5D.close(did)); %#ok<NASGU>
    refs=H5D.read(did);refs=reshape(refs,8,[]);
    assert(channel<=size(refs,2),'Requested channel is unavailable.');
    target=H5R.dereference(fid,'H5R_OBJECT',refs(:,channel));
    closeTarget=onCleanup(@() H5O.close(target)); %#ok<NASGU>
    path=H5I.get_name(target);
end

function values=readVectorRange(file,path,first,n)
    info=h5info(file,path);dims=info.Dataspace.Size;
    assert(sum(dims>1)<=1&&prod(dims)>=first+n-1,'Invalid or short epoch vector.');
    axis=find(dims>1,1);if isempty(axis),axis=1;end
    start=ones(1,numel(dims));count=ones(1,numel(dims));
    start(axis)=first;count(axis)=n;
    values=h5read(file,path,start,count);values=values(:);
end

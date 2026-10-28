function results = sqm_20260922_delt_wscm(userCfg)
%SQM_20260922_DELT_WSCM  Delta/WSCM SQM comparison for spoofing and multipath.
%
% This file is intentionally independent of the supplied source file.  It
% keeps only the two requested detection statistics:
%   Delta = (I_E - I_L)/(2 I_P)
%   WSCM  = weighted early/late correlator-power difference
%
% The following choices are explicit in defaultConfig(), so they can be
% changed without editing the detection logic:
%   - threshold-calibration intervals for spoofing and multipath;
%   - clean-test and event-test intervals;
%   - feature-plot intervals;
%   - requested early correlator positions;
%   - WSCM symmetric offsets, sigma multiplier and smoothing mode.
%
% Example:
%   R = sqm_20260922_delt_wscm();
%   R = sqm_20260922_delt_wscm(struct( ...
%       'spoofCorrelatorPositionChips',-0.7, ...
%       'multipathCorrelatorPositionChips',-1.5, ...
%       'spoofThresholdTimeSec',[20 80], ...
%       'multipathThresholdTimeSec',[20 60]));
%
% The combined figure is saved as PNG/PDF/FIG.  In the multipath panel,
% post-event alarms are deliberately labelled false alarms: multipath is a
% negative class when the detector is intended to detect spoofing.

if nargin < 1 || isempty(userCfg)
    userCfg = struct();
end

cfg = mergeConfig(defaultConfig(),userCfg);
validateConfig(cfg);
if ~exist(cfg.outputDir,'dir')
    mkdir(cfg.outputDir);
end

% Cache only raw correlator data.  Thresholds, taps and smoothing are not
% cached, so changing any analysis parameter immediately takes effect.
persistent rawSpoof rawMultipath cacheKey
if cfg.forceReload
    rawSpoof = [];
    rawMultipath = [];
    cacheKey = [];
end

key = makeCacheKey(cfg);
if isempty(rawSpoof) || isempty(rawMultipath) || isempty(cacheKey) || ...
        ~sameCacheKey(cacheKey,key)
    fprintf('Loading spoofing correlator data...\n');
    rawSpoof = loadScenario(cfg.spoofFile,cfg.spoofChannel, ...
        cfg.spoofEndTime,cfg.numCorrelators,cfg.fs,cfg.requireCorrFingers);
    fprintf('Loading multipath correlator data...\n');
    rawMultipath = loadScenario(cfg.multipathFile,cfg.multipathChannel, ...
        cfg.multipathEndTime,cfg.numCorrelators,cfg.fs,cfg.requireCorrFingers);
    cacheKey = key;
else
    fprintf('Reusing cached raw correlator data.\n');
end

spoof = calculateScenarioStatistics(rawSpoof,cfg.spoofWindowSec, ...
    cfg.spoofCorrelatorPositionChips,cfg);
multi = calculateScenarioStatistics(rawMultipath,cfg.multipathWindowSec, ...
    cfg.multipathCorrelatorPositionChips,cfg);

methodNames = {'Delta';'WSCM'};

% Every sample-selection mask is explicit.  The event masks are strictly
% after the split time so the boundary sample cannot be counted twice.
spoofCalib = intervalMask(spoof.timeSec,cfg.spoofThresholdTimeSec,true,true) & ...
    spoof.timeSec <= cfg.spoofSplitTime;
multiCalib = intervalMask(multi.timeSec,cfg.multipathThresholdTimeSec,true,true) & ...
    multi.timeSec <= cfg.multipathSplitTime;
spoofCleanTest = intervalMask(spoof.timeSec,cfg.spoofCleanTestTimeSec,true,true) & ...
    spoof.timeSec <= cfg.spoofSplitTime & ~spoofCalib;
multiCleanTest = intervalMask(multi.timeSec,cfg.multipathCleanTestTimeSec,true,true) & ...
    multi.timeSec <= cfg.multipathSplitTime & ~multiCalib;
spoofEventTest = intervalMask(spoof.timeSec,cfg.spoofEventTestTimeSec,true,true) & ...
    spoof.timeSec > cfg.spoofSplitTime;
multiEventTest = intervalMask(multi.timeSec,cfg.multipathEventTestTimeSec,true,true) & ...
    multi.timeSec > cfg.multipathSplitTime;

assert(nnz(spoofCalib) >= cfg.minimumThresholdSamples && ...
    nnz(multiCalib) >= cfg.minimumThresholdSamples && ...
    nnz(spoofCleanTest)>0 && nnz(multiCleanTest)>0 && ...
    nnz(spoofEventTest)>0 && nnz(multiEventTest)>0, ...
    'One or more configured sample intervals is empty or too short.');

nMethods = numel(methodNames);
spoofThresholds = nan(nMethods,2);
multiThresholds = nan(nMethods,2);
spoofCentres = nan(nMethods,1);
multiCentres = nan(nMethods,1);
spoofSigmas = nan(nMethods,1);
multiSigmas = nan(nMethods,1);
decisionsSpoof = false(numel(spoof.timeSec),nMethods);
decisionsMulti = false(numel(multi.timeSec),nMethods);
scoresSpoof = nan(numel(spoof.timeSec),nMethods);
scoresMulti = nan(numel(multi.timeSec),nMethods);

for m = 1:nMethods
    [spoofCentres(m),spoofSigmas(m),spoofThresholds(m,:)] = ...
        estimateThreshold(spoof.statistics(:,m),spoofCalib,cfg);
    [multiCentres(m),multiSigmas(m),multiThresholds(m,:)] = ...
        estimateThreshold(multi.statistics(:,m),multiCalib,cfg);
    decisionsSpoof(:,m) = outsideThreshold(spoof.statistics(:,m), ...
        spoofThresholds(m,:));
    decisionsMulti(:,m) = outsideThreshold(multi.statistics(:,m), ...
        multiThresholds(m,:));
    scoresSpoof(:,m) = abs(spoof.statistics(:,m)-spoofCentres(m)) ./ spoofSigmas(m);
    scoresMulti(:,m) = abs(multi.statistics(:,m)-multiCentres(m)) ./ multiSigmas(m);
end

metricRows = cell(3*nMethods,15);
row = 0;
for m = 1:nMethods
    row = row + 1;
    metricRows(row,:) = makeMetricRow(methodNames{m},'Spoofing', ...
        spoofCleanTest,spoofEventTest,decisionsSpoof(:,m),scoresSpoof(:,m), ...
        spoofCalib,true);
    row = row + 1;
    metricRows(row,:) = makeMetricRow(methodNames{m},'Multipath', ...
        multiCleanTest,multiEventTest,decisionsMulti(:,m),scoresMulti(:,m), ...
        multiCalib,false);
    row = row + 1;
    metricRows(row,:) = makeCombinedMetricRow(methodNames{m}, ...
        spoofCleanTest,spoofEventTest,decisionsSpoof(:,m),scoresSpoof(:,m), ...
        multiCleanTest,multiEventTest,decisionsMulti(:,m),scoresMulti(:,m), ...
        spoofCalib,multiCalib);
end
metrics = cell2table(metricRows,'VariableNames', ...
    {'Method','Scenario','Accuracy','Precision','Recall','F1','AUC', ...
     'CleanFalseAlarmRate','EventAlarmRate','TP','TN','FP','FN', ...
     'TestCount','CalibrationCount'});

thresholdTable = [thresholdRows(methodNames,'Spoofing',spoofCentres, ...
    spoofSigmas,spoofThresholds,spoofCalib,cfg.spoofThresholdTimeSec, ...
    spoof.tapInfo); ...
    thresholdRows(methodNames,'Multipath',multiCentres,multiSigmas, ...
    multiThresholds,multiCalib,cfg.multipathThresholdTimeSec, ...
    multi.tapInfo)];

spoofTable = makeSeriesTable(spoof,decisionsSpoof,spoofCalib, ...
    spoofCleanTest,spoofEventTest,methodNames);
multiTable = makeSeriesTable(multi,decisionsMulti,multiCalib, ...
    multiCleanTest,multiEventTest,methodNames);

writetable(metrics,fullfile(cfg.outputDir,'sqm_delt_wscm_metrics.csv'));
writetable(thresholdTable,fullfile(cfg.outputDir,'sqm_delt_wscm_thresholds.csv'));
writetable(spoofTable,fullfile(cfg.outputDir,'sqm_delt_wscm_spoofing_series.csv'));
writetable(multiTable,fullfile(cfg.outputDir,'sqm_delt_wscm_multipath_series.csv'));

results = struct();
results.config = cfg;
results.methods = methodNames;
results.metrics = metrics;
results.thresholds = thresholdTable;
results.spoofing = spoofTable;
results.multipath = multiTable;
results.scores = struct('spoofing',scoresSpoof,'multipath',scoresMulti);
results.tapInfo = struct('spoofing',spoof.tapInfo,'multipath',multi.tapInfo);
results.wscmInfo = struct('spoofing',spoof.wscmInfo,'multipath',multi.wscmInfo);
results.metricDefinition = ['Spoofing post-event samples are positive; clean and ' ...
    'multipath post-event samples are negative. Therefore, alarms in the ' ...
    'multipath event interval are false alarms for a spoofing detector.'];
save(fullfile(cfg.outputDir,'sqm_20260922_delt_wscm_results.mat'), ...
    'results','-v7.3');

if cfg.saveFigures
    saveFigures(results,cfg);
end

fprintf('\nDelta/WSCM SQM results:\n');
disp(metrics);
fprintf('Outputs: %s\n',cfg.outputDir);
end

%% --------------------------- configuration ---------------------------
function cfg = defaultConfig()
rootDir = fileparts(mfilename('fullpath'));
cfg = struct();


%% %静态场景
% cfg.spoofFile = 'data/raw/asy_tim_0.1v_81corr.mat';
% cfg.multipathFile = 'data/raw/multipath_static_1_corr81.mat';
% cfg.spoofChannel = 7;
% cfg.multipathChannel = 3;
% cfg.spoofStartTime = 20;
% cfg.spoofSplitTime = 103;
% cfg.spoofEndTime = 180;
% cfg.multipathStartTime = 20;
% cfg.multipathSplitTime = 83.5;
% cfg.multipathEndTime = 140;
% cfg.fs = 1000;
% cfg.numCorrelators = 81;
%
% % -------------------- tune sample selection here ---------------------
% % Threshold calibration samples: clean data only, before the event.
% cfg.spoofThresholdTimeSec = [60 80];
% cfg.multipathThresholdTimeSec = [65 75];
% % Held-out clean samples used to estimate the clean false-alarm rate.
% cfg.spoofCleanTestTimeSec = [30 50];
% cfg.multipathCleanTestTimeSec = [30 50];
% % Event samples used to estimate spoofing detection / multipath false alarms.
% % cfg.spoofEventTestTimeSec = [103 113];
% % cfg.multipathEventTestTimeSec = [84 93];
%
% cfg.spoofEventTestTimeSec = [103 123];
% cfg.multipathEventTestTimeSec = [84 104];
% % Time ranges shown in the combined intermediate-feature figure.
% cfg.spoofFeaturePlotTimeSec = [87 127];
% cfg.multipathFeaturePlotTimeSec = [63 103];
%
% % A negative value denotes the requested early tap.  The late tap is its
% % symmetric positive counterpart.  The actual nearest grid positions are
% % saved in results.tapInfo and the threshold table.
% cfg.spoofCorrelatorPositionChips = -0.5;
% % cfg.multipathCorrelatorPositionChips = -1.7;
% cfg.multipathCorrelatorPositionChips = -1.2;
% % cfg.wscmOffsetsChips = [0.2 0.4 0.6 0.8 1.0];
% cfg.wscmOffsetsChips = [0.4 0.6 0.8 1.2 1.5];
%
%
% cfg.spoofWindowSec = 0.5;
% cfg.multipathWindowSec = 0.2;
% % Causal smoothing is appropriate for online detection.  Set to 'centered'
% % only when reproducing the legacy offline centered-window curves.
% cfg.smoothingMode = 'causal';
% cfg.sigmaMultiplier = 3;
% cfg.minimumThresholdSamples = 20;
% cfg.requireCorrFingers = true;
% cfg.forceReload = false;
% cfg.saveFigures = true;
% cfg.figureVisible = 'off';
% cfg.closeFiguresAfterSave = true;
% cfg.outputDir = fullfile(rootDir,'results_sqm_delt_wscm_static');
%


%% 动态场景
cfg.spoofFile = 'data/raw/asy_tim_0.5v_81corr.mat';
cfg.multipathFile = 'data/raw/multipath_dy_15m_corr81.mat';
cfg.spoofChannel = 7;
cfg.multipathChannel = 5;
cfg.spoofStartTime = 20;
cfg.spoofSplitTime = 79;
cfg.spoofEndTime = 180;
cfg.multipathStartTime = 20;
cfg.multipathSplitTime = 80.5;
cfg.multipathEndTime = 140;
cfg.fs = 1000;
cfg.numCorrelators = 81;

% -------------------- tune sample selection here ---------------------
% Threshold calibration samples: clean data only, before the event.
cfg.spoofThresholdTimeSec = [60 70];
cfg.multipathThresholdTimeSec = [60 70];
% Held-out clean samples used to estimate the clean false-alarm rate.
cfg.spoofCleanTestTimeSec = [30 50];
cfg.multipathCleanTestTimeSec = [30 50];
% Event samples used to estimate spoofing detection / multipath false alarms.
% cfg.spoofEventTestTimeSec = [103 113];
% cfg.multipathEventTestTimeSec = [84 93];

cfg.spoofEventTestTimeSec = [79 99];
cfg.multipathEventTestTimeSec = [84 104];
% Time ranges shown in the combined intermediate-feature figure.
cfg.spoofFeaturePlotTimeSec = [60 110];
cfg.multipathFeaturePlotTimeSec = [50 110];

% A negative value denotes the requested early tap.  The late tap is its
% symmetric positive counterpart.  The actual nearest grid positions are
% saved in results.tapInfo and the threshold table.
% cfg.spoofCorrelatorPositionChips = -0.5;
cfg.spoofCorrelatorPositionChips = -1.2;
% cfg.multipathCorrelatorPositionChips = -1.7;
cfg.multipathCorrelatorPositionChips = -1.2;
% cfg.wscmOffsetsChips = [0.2 0.4 0.6 0.8 1.0];
cfg.wscmOffsetsChips = [0.4 0.6 0.8 1.2 1.5];


cfg.spoofWindowSec = 0.5;
cfg.multipathWindowSec = 0.2;
% Causal smoothing is appropriate for online detection.  Set to 'centered'
% only when reproducing the legacy offline centered-window curves.
cfg.smoothingMode = 'causal';
cfg.sigmaMultiplier = 3;
cfg.minimumThresholdSamples = 20;
cfg.requireCorrFingers = true;
cfg.forceReload = false;
cfg.saveFigures = true;
cfg.figureVisible = 'off';
cfg.closeFiguresAfterSave = true;
cfg.outputDir = fullfile(rootDir,'results_sqm_delt_wscm_dynamic');



end

function cfg = mergeConfig(cfg,userCfg)
names = fieldnames(userCfg);
for k = 1:numel(names)
    if ~isfield(cfg,names{k})
        error('Unknown configuration field: %s',names{k});
    end
    cfg.(names{k}) = userCfg.(names{k});
end
end

function validateConfig(cfg)
assert(isfile(cfg.spoofFile),'Spoofing file not found: %s',cfg.spoofFile);
assert(isfile(cfg.multipathFile),'Multipath file not found: %s',cfg.multipathFile);
assert(isscalar(cfg.fs) && isfinite(cfg.fs) && cfg.fs>0);
assert(isscalar(cfg.numCorrelators) && cfg.numCorrelators==81, ...
    'This implementation expects the complete 81-correlator profile.');
assert(cfg.spoofStartTime < cfg.spoofSplitTime && ...
    cfg.spoofSplitTime < cfg.spoofEndTime);
assert(cfg.multipathStartTime < cfg.multipathSplitTime && ...
    cfg.multipathSplitTime < cfg.multipathEndTime);
assert(cfg.spoofWindowSec>0 && cfg.multipathWindowSec>0);
assert(isscalar(cfg.sigmaMultiplier) && isfinite(cfg.sigmaMultiplier) && ...
    cfg.sigmaMultiplier>0);
assert(isscalar(cfg.minimumThresholdSamples) && ...
    cfg.minimumThresholdSamples>=2 && mod(cfg.minimumThresholdSamples,1)==0);
assert(islogical(cfg.forceReload) && isscalar(cfg.forceReload));
assert(islogical(cfg.requireCorrFingers) && isscalar(cfg.requireCorrFingers));
assert(any(strcmpi(cfg.smoothingMode,{'causal','centered'})), ...
    'smoothingMode must be causal or centered.');
assert(any(strcmpi(cfg.figureVisible,{'on','off'})));
assert(islogical(cfg.saveFigures) && isscalar(cfg.saveFigures));
assert(islogical(cfg.closeFiguresAfterSave) && isscalar(cfg.closeFiguresAfterSave));
assert(isfinite(cfg.spoofCorrelatorPositionChips) && ...
    cfg.spoofCorrelatorPositionChips<0);
assert(isfinite(cfg.multipathCorrelatorPositionChips) && ...
    cfg.multipathCorrelatorPositionChips<0);
assert(isnumeric(cfg.wscmOffsetsChips) && ~isempty(cfg.wscmOffsetsChips) && ...
    all(isfinite(cfg.wscmOffsetsChips)) && all(cfg.wscmOffsetsChips>0));

rangeNames = {'spoofThresholdTimeSec','multipathThresholdTimeSec', ...
    'spoofCleanTestTimeSec','multipathCleanTestTimeSec', ...
    'spoofEventTestTimeSec','multipathEventTestTimeSec', ...
    'spoofFeaturePlotTimeSec','multipathFeaturePlotTimeSec'};
for k = 1:numel(rangeNames)
    r = cfg.(rangeNames{k});
    assert(isnumeric(r) && numel(r)==2 && all(isfinite(r)) && r(1)<r(2), ...
        '%s must be a finite increasing [start end] vector.',rangeNames{k});
end
assert(cfg.spoofThresholdTimeSec(1)>=cfg.spoofStartTime && ...
    cfg.spoofThresholdTimeSec(2)<=cfg.spoofSplitTime);
assert(cfg.multipathThresholdTimeSec(1)>=cfg.multipathStartTime && ...
    cfg.multipathThresholdTimeSec(2)<=cfg.multipathSplitTime);
assert(cfg.spoofCleanTestTimeSec(1)>=cfg.spoofStartTime && ...
    cfg.spoofCleanTestTimeSec(2)<=cfg.spoofSplitTime);
assert(cfg.multipathCleanTestTimeSec(1)>=cfg.multipathStartTime && ...
    cfg.multipathCleanTestTimeSec(2)<=cfg.multipathSplitTime);
assert(cfg.spoofEventTestTimeSec(1)>=cfg.spoofSplitTime && ...
    cfg.spoofEventTestTimeSec(2)<=cfg.spoofEndTime);
assert(cfg.multipathEventTestTimeSec(1)>=cfg.multipathSplitTime && ...
    cfg.multipathEventTestTimeSec(2)<=cfg.multipathEndTime);
assert(cfg.spoofFeaturePlotTimeSec(1)>=cfg.spoofStartTime && ...
    cfg.spoofFeaturePlotTimeSec(2)<=cfg.spoofEndTime);
assert(cfg.multipathFeaturePlotTimeSec(1)>=cfg.multipathStartTime && ...
    cfg.multipathFeaturePlotTimeSec(2)<=cfg.multipathEndTime);
end

%% ----------------------------- data I/O -------------------------------
function key = makeCacheKey(cfg)
key = struct('spoofFile',cfg.spoofFile,'multipathFile',cfg.multipathFile, ...
    'spoofChannel',cfg.spoofChannel,'multipathChannel',cfg.multipathChannel, ...
    'spoofEndTime',cfg.spoofEndTime,'multipathEndTime',cfg.multipathEndTime, ...
    'numCorrelators',cfg.numCorrelators,'fs',cfg.fs, ...
    'requireCorrFingers',cfg.requireCorrFingers);
end

function tf = sameCacheKey(a,b)
tf = strcmp(a.spoofFile,b.spoofFile) && strcmp(a.multipathFile,b.multipathFile) && ...
    a.spoofChannel==b.spoofChannel && a.multipathChannel==b.multipathChannel && ...
    a.spoofEndTime==b.spoofEndTime && a.multipathEndTime==b.multipathEndTime && ...
    a.numCorrelators==b.numCorrelators && a.fs==b.fs && ...
    a.requireCorrFingers==b.requireCorrFingers;
end

function out = loadScenario(fileName,channelIdx,endTime,nCorr,fs,requireCorrFingers)
s = load(fileName,'trackData');
assert(isfield(s,'trackData'),'trackData is missing from %s',fileName);
assert(isfield(s.trackData,'beib1') && ...
    numel(s.trackData.beib1.channel)>=channelIdx, ...
    'Channel %d is not available in %s.',channelIdx,fileName);
ch = s.trackData.beib1.channel(channelIdx);
assert(isfield(ch,'I_P'),'I_P is missing from %s, channel %d.',fileName,channelIdx);
lastIdx = min(numel(ch.I_P),floor(endTime*fs)+1);
assert(lastIdx>=2,'Not enough samples in %s.',fileName);
idx = (1:lastIdx).';

out.timeSec = (idx-1)/fs;
out.U = extractCorrelatorIQ(ch,idx,nCorr);
out.offsets = correlatorOffsets(ch,nCorr,requireCorrFingers);
out.sourceFile = fileName;
out.channel = channelIdx;
end

function U = extractCorrelatorIQ(ch,idx,nCorr)
earlyI = cell(1,40); earlyQ = cell(1,40);
lateI = cell(1,40); lateQ = cell(1,40);
for j = 1:40
    ke = 40-j;
    if ke==0
        earlyI{j}='I_E'; earlyQ{j}='Q_E';
    else
        earlyI{j}=sprintf('I_E_%d',ke);
        earlyQ{j}=sprintf('Q_E_%d',ke);
    end
    kl = j-1;
    if kl==0
        lateI{j}='I_L'; lateQ{j}='Q_L';
    else
        lateI{j}=sprintf('I_L_%d',kl);
        lateQ{j}=sprintf('Q_L_%d',kl);
    end
end
allI = [earlyI,{'I_P'},lateI];
allQ = [earlyQ,{'Q_P'},lateQ];
for k = 1:nCorr
    assert(isfield(ch,allI{k}) && isfield(ch,allQ{k}), ...
        'Missing 81-correlator field: %s or %s.',allI{k},allQ{k});
end
U = complex(zeros(nCorr,numel(idx)));
for k = 1:nCorr
    U(k,:) = double(ch.(allI{k})(idx)).' + ...
        1i*double(ch.(allQ{k})(idx)).';
end
end

function offsets = correlatorOffsets(ch,nCorr,requireCorrFingers)
assert(isfield(ch,'corrFingers') && numel(ch.corrFingers)==nCorr, ...
    'corrFingers must contain the %d-point correlator grid.',nCorr);
offsets = double(ch.corrFingers(:));
if any(~isfinite(offsets)) || any(diff(offsets)<=0)
    if requireCorrFingers
        error('corrFingers is not a finite strictly increasing grid.');
    end
    offsets = linspace(-2,2,nCorr).';
end
end

%% ----------------------------- features ------------------------------
function out = calculateScenarioStatistics(raw,windowSec,requestedEarly,cfg)
[~,earlyIdx] = min(abs(raw.offsets-requestedEarly));
[~,lateIdx] = min(abs(raw.offsets+requestedEarly));
[~,promptIdx] = min(abs(raw.offsets));
actualEarly = raw.offsets(earlyIdx);
actualLate = raw.offsets(lateIdx);
assert(actualEarly<0 && actualLate>0, ...
    'The selected Delta taps must straddle the prompt correlator.');

IE = real(raw.U(earlyIdx,:)).';
IL = real(raw.U(lateIdx,:)).';
IP = real(raw.U(promptIdx,:)).';
deltaRaw = (IE-IL)./(2*IP);
deltaRaw(~isfinite(deltaRaw)) = NaN;

[wscmRaw,wscmInfo] = calculateWSCM(raw.U,raw.offsets,cfg.wscmOffsetsChips);
wscmRaw(~isfinite(wscmRaw)) = NaN;

out = raw;
out.featureRaw = [deltaRaw,wscmRaw];
out.statistics = [windowedMean(deltaRaw,windowSec,cfg.fs,cfg.smoothingMode), ...
                  windowedMean(wscmRaw,windowSec,cfg.fs,cfg.smoothingMode)];
out.wscmInfo = wscmInfo;
out.tapInfo = struct('requestedEarlyChips',requestedEarly, ...
    'actualEarlyChips',actualEarly,'actualLateChips',actualLate, ...
    'promptChips',raw.offsets(promptIdx),'earlyIndex',earlyIdx, ...
    'lateIndex',lateIdx,'promptIndex',promptIdx);
end

function y = windowedMean(x,windowSec,fs,mode)
x = double(x(:));
n = numel(x);
w = max(1,round(windowSec*fs));
y = nan(n,1);
for i = 1:n
    switch lower(mode)
        case 'causal'
            a = max(1,i-w+1);
            b = i;
        case 'centered'
            half = floor((w-1)/2);
            a = max(1,i-half);
            b = min(n,i+(w-1-half));
        otherwise
            error('Unsupported smoothing mode: %s',mode);
    end
    values = x(a:b);
    if any(isfinite(values))
        y(i) = mean(values,'omitnan');
    end
end
end

function [d,info] = calculateWSCM(U,offsets,requested)
requested = sort(double(requested(:)));
n = numel(requested);
e = zeros(n,1); l = zeros(n,1); actual = zeros(n,1);
for k = 1:n
    [~,e(k)] = min(abs(offsets+requested(k)));
    [~,l(k)] = min(abs(offsets-requested(k)));
    actual(k) = 0.5*(abs(offsets(e(k)))+abs(offsets(l(k))));
end
assert(numel(unique(e))==n && numel(unique(l))==n && all(actual>0), ...
    'WSCM offsets map to duplicate or invalid correlator taps.');
% For this implementation alpha_i is proportional to 1/d_i.  The common
% noise-variance factor cancels when the weights are normalized.
alpha = (1./actual)/sum(1./actual);
p = abs(U).^2;
early = alpha.'*p(e,:);
late = alpha.'*p(l,:);
d = (early-late).';
info = struct('requestedOffsetsChips',requested, ...
    'actualOffsetsChips',actual,'weights',alpha, ...
    'earlyIndices',e,'lateIndices',l);
end

%% ------------------------- sample/threshold --------------------------
function mask = intervalMask(timeSec,r,leftClosed,rightClosed)
if leftClosed, leftOK = timeSec>=r(1); else, leftOK = timeSec>r(1); end
if rightClosed, rightOK = timeSec<=r(2); else, rightOK = timeSec<r(2); end
mask = leftOK & rightOK;
end

function [centre,sigma,threshold] = estimateThreshold(x,mask,cfg)
values = double(x(mask));
values = values(isfinite(values));
assert(numel(values)>=cfg.minimumThresholdSamples, ...
    'Too few finite calibration samples: %d.',numel(values));
centre = mean(values);
sigma = std(values,0);
if ~isfinite(sigma) || sigma<=eps(max(1,abs(centre)))
    sigma = eps(max(1,abs(centre)));
end
threshold = centre + cfg.sigmaMultiplier*sigma*[-1 1];
end

function tf = outsideThreshold(x,t)
tf = isfinite(x) & (x<t(1) | x>t(2));
end

function row = makeMetricRow(method,scenario,cleanMask,eventMask, ...
    decision,score,calibrationMask,eventIsPositive)
evalMask = cleanMask | eventMask;
labels = false(size(evalMask));
if eventIsPositive
    labels(eventMask) = true;
end
[accuracy,precision,recall,f1,auc,tp,tn,fp,fn] = binaryMetrics( ...
    labels(evalMask),decision(evalMask),score(evalMask));
cleanFalseAlarmRate = mean(double(decision(cleanMask)));
eventAlarmRate = mean(double(decision(eventMask)));
row = {method,scenario,accuracy,precision,recall,f1,auc, ...
    cleanFalseAlarmRate,eventAlarmRate,tp,tn,fp,fn,nnz(evalMask), ...
    nnz(calibrationMask)};
end

function row = makeCombinedMetricRow(method,spoofClean,spoofEvent, ...
    spoofDecision,spoofScore,multiClean,multiEvent,multiDecision, ...
    multiScore,spoofCalib,multiCalib)
labels = [false(nnz(spoofClean),1); true(nnz(spoofEvent),1); ...
    false(nnz(multiClean),1); false(nnz(multiEvent),1)];
decision = [spoofDecision(spoofClean); spoofDecision(spoofEvent); ...
    multiDecision(multiClean); multiDecision(multiEvent)];
score = [spoofScore(spoofClean); spoofScore(spoofEvent); ...
    multiScore(multiClean); multiScore(multiEvent)];
[accuracy,precision,recall,f1,auc,tp,tn,fp,fn] = binaryMetrics( ...
    labels,decision,score);
cleanFalseAlarmRate = mean(double([spoofDecision(spoofClean); ...
    multiDecision(multiClean)]));
row = {method,'Combined',accuracy,precision,recall,f1,auc, ...
    cleanFalseAlarmRate,NaN,tp,tn,fp,fn,numel(labels), ...
    nnz(spoofCalib)+nnz(multiCalib)};
end

function [accuracy,precision,recall,f1,auc,tp,tn,fp,fn] = binaryMetrics( ...
    labels,decision,score)
labels = logical(labels(:));
decision = logical(decision(:));
score = double(score(:));
tp = nnz(decision & labels);
tn = nnz(~decision & ~labels);
fp = nnz(decision & ~labels);
fn = nnz(~decision & labels);
accuracy = (tp+tn)/max(1,numel(labels));
if nnz(labels)==0 || nnz(~labels)==0
    precision = NaN; recall = NaN; f1 = NaN;
else
    precision = safeDivide(tp,tp+fp);
    recall = safeDivide(tp,tp+fn);
    f1 = safeDivide(2*precision*recall,precision+recall);
end
auc = binaryAuc(labels,score);
end

function value = safeDivide(a,b)
if b==0, value = 0; else, value = a/b; end
end

function auc = binaryAuc(labels,score)
valid = isfinite(score);
labels = labels(valid); score = score(valid);
nPos = nnz(labels); nNeg = nnz(~labels);
if nPos==0 || nNeg==0
    auc = NaN;
    return;
end
[sortedScore,order] = sort(score);
sortedLabels = labels(order);
n = numel(sortedScore); ranks = zeros(n,1); i = 1;
while i<=n
    j = i;
    while j<n && sortedScore(j+1)==sortedScore(i), j = j+1; end
    ranks(i:j) = (i+j)/2;
    i = j+1;
end
auc = (sum(ranks(sortedLabels))-nPos*(nPos+1)/2)/(nPos*nNeg);
end

function T = thresholdRows(names,scenario,centres,sigmas,thresholds, ...
    calibMask,selectionRange,tapInfo)
n = numel(names);
T = table(names,repmat({scenario},n,1),centres,sigmas, ...
    thresholds(:,1),thresholds(:,2),repmat(nnz(calibMask),n,1), ...
    repmat(selectionRange(1),n,1),repmat(selectionRange(2),n,1), ...
    repmat(tapInfo.requestedEarlyChips,n,1), ...
    repmat(tapInfo.actualEarlyChips,n,1), ...
    'VariableNames',{'Method','Scenario','CleanMean','CleanStd', ...
    'LowerThreshold','UpperThreshold','CalibrationCount', ...
    'ThresholdStartSec','ThresholdEndSec','RequestedEarlyChips', ...
    'ActualEarlyChips'});
end

%% ------------------------------ outputs -------------------------------
function T = makeSeriesTable(S,dec,calibMask,cleanTestMask,eventTestMask,names)
T = table(S.timeSec,repmat(S.tapInfo.actualEarlyChips,numel(S.timeSec),1), ...
    calibMask,cleanTestMask,eventTestMask, ...
    'VariableNames',{'TimeSec','ActualEarlyCorrelatorChips', ...
    'CalibrationClean','CleanTestSet','EventTestSet'});
for k = 1:numel(names)
    T.([names{k} '_Statistic']) = S.statistics(:,k);
    T.([names{k} '_Decision']) = dec(:,k);
end
end

function saveFigures(R,cfg)
plotCombinedFeatureComparison(R,cfg);

M = R.metrics(strcmp(R.metrics.Scenario,'Combined'),:);
f = figure('Visible',cfg.figureVisible,'Color','w', ...
    'Position',[100 100 1000 500]);
vals = [M.Accuracy,M.F1,M.AUC];
b = bar(categorical(M.Method,M.Method),vals,'grouped');
b(1).FaceColor=[0.16 0.35 0.58];
b(2).FaceColor=[0.82 0.55 0.18];
b(3).FaceColor=[0.23 0.55 0.39];
ylim([0 1]); ylabel('Score'); grid on; box on;
legend({'Accuracy','F1','AUC'},'Location','southoutside', ...
    'Orientation','horizontal','Box','off');
title('Combined Delta/WSCM SQM metrics');
saveFigureFiles(f,fullfile(cfg.outputDir,'sqm_delt_wscm_summary'));
if cfg.closeFiguresAfterSave, close(f); end
end

function plotCombinedFeatureComparison(R,cfg)
names = R.methods;
data = {R.spoofing,R.multipath};
thresholds = {R.thresholds(strcmp(R.thresholds.Scenario,'Spoofing'),:), ...
    R.thresholds(strcmp(R.thresholds.Scenario,'Multipath'),:)};
scenarioNames = {'Spoofing','Multipath'};
plotRanges = {cfg.spoofFeaturePlotTimeSec,cfg.multipathFeaturePlotTimeSec};
splitTimes = [cfg.spoofSplitTime,cfg.multipathSplitTime];

f = figure('Visible',cfg.figureVisible,'Color','w', ...
    'Position',[60 60 1400 760]);
tl = tiledlayout(numel(names),2,'TileSpacing','compact','Padding','compact');
for m = 1:numel(names)
    for s = 1:2
        ax = nexttile(tl); hold(ax,'on');
        T = data{s}; Q = thresholds{s};
        r = plotRanges{s}; split = splitTimes(s);
        viewMask = T.TimeSec>=r(1) & T.TimeSec<=r(2);
        z = T.([names{m} '_Statistic']);
        pre = viewMask & T.TimeSec<=split;
        post = viewMask & T.TimeSec>split;
        plot(ax,T.TimeSec(pre),z(pre),'Color',[0.13 0.35 0.66], ...
            'LineWidth',1.0,'DisplayName','Before event');
        plot(ax,T.TimeSec(post),z(post),'Color',[0.92 0.47 0.13], ...
            'LineWidth',1.0,'DisplayName','After event');
        yline(ax,Q.UpperThreshold(m),'--','Color',[0.78 0.20 0.20], ...
            'LineWidth',1.1,'DisplayName','Threshold');
        yline(ax,Q.LowerThreshold(m),'--','Color',[0.78 0.20 0.20], ...
            'LineWidth',1.1,'HandleVisibility','off');
        alarm = viewMask & T.([names{m} '_Decision']) & T.EventTestSet;
        scatter(ax,T.TimeSec(alarm),z(alarm),16,[0.80 0.05 0.05], ...
            'filled','DisplayName','Alarm samples');
        xline(ax,split,':k','LineWidth',1.0,'DisplayName','Event onset');
        xlim(ax,r); grid(ax,'on'); box(ax,'on');
        title(ax,sprintf('%s | %s',scenarioNames{s},names{m}));
        if m==numel(names), xlabel(ax,'Source time (s)'); end
        if s==1, ylabel(ax,'Detection statistic'); end
        decisionForPlot = T.([names{m} '_Decision']);
        alarmRate = mean(double(decisionForPlot(T.EventTestSet)));
        text(ax,0.02,0.94,sprintf('event alarm rate = %.1f%%',100*alarmRate), ...
            'Units','normalized','FontSize',9,'Color',[0.25 0.25 0.25]);
        if m==1 && s==1
            legend(ax,'Location','best','Box','off');
        end
    end
end
title(tl,['Intermediate features: event distortion and threshold crossing (' ...
    'multipath alarms are false alarms)']);
saveFigureFiles(f,fullfile(cfg.outputDir,'sqm_spoof_multipath_feature_comparison'));
if cfg.closeFiguresAfterSave, close(f); end
end

function saveFigureFiles(f,baseName)
savefig(f,[baseName '.fig']);
exportgraphics(f,[baseName '.png'],'Resolution',300);
exportgraphics(f,[baseName '.pdf'],'ContentType','vector');
end

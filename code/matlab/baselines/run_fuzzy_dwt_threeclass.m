function results = run_fuzzy_dwt_threeclass(userCfg)
%RUN_FUZZY_DWT_THREECLASS
% DWT-fuzzy three-class GNSS detection based on Tohidi & Mosavi (2024).
%
% 论文给出的流程是两级分类，而不是一个单级三分类器：
%   Stage 1: 全部 DWT 系数的均值/方差 -> clean 或 interference；
%   Stage 2: selected coefficients -> multipath or spoofing.
% Current project preset: [17 18 20 21 27 38 39 40]; see defaultConfig.
%
% Gaussian membership functions are fitted to training features for each class.
% Decisions use maximum joint membership in the two-stage classifier.
% Labels: 0=authentic, 1=spoofing, 2=multipath.
%
% 输入相关峰为每历元 81 点 |I+jQ|。DWT 使用 sym4、4 层分解；默认采用因果平滑，
% 防止测试时刻使用未来信息。程序同时输出单历元原生结果和与深度网络一致的
% 10 ms 非重叠窗口结果。
%
% 用法：
%   results = run_fuzzy_dwt_threeclass();
%   cfg = struct('useFullRecording',false,'preEventMs',1000, ...
%       'postEventMs',1000,'saveFigures',false);
%   results = run_fuzzy_dwt_threeclass(cfg);

if nargin < 1
    userCfg = struct();
end
cfg = defaultConfig();
cfg = mergeConfig(cfg,userCfg);
validateConfig(cfg);
if ~exist(cfg.outputDir,'dir'), mkdir(cfg.outputDir); end
rng(cfg.randomSeed,'twister');

assert(exist('wavedec','file')==2, ...
    'Wavelet Toolbox is required: wavedec with sym4 is part of the paper method.');
dwtmode('sym','nodisp');

fprintf('Extracting spoofing DWT features...\n');
spoof = loadAndExtractScenario(cfg.spoofFile,cfg.spoofChannel,cfg.spoofOnsetMs, ...
    cfg.preEventMs,cfg.postEventMs,1,1,cfg.spoofCorrelatorPositionChips,cfg);
fprintf('Extracting multipath DWT features...\n');
multi = loadAndExtractScenario(cfg.multipathFile,cfg.multipathChannel,cfg.multipathOnsetMs, ...
    cfg.preEventMs,cfg.postEventMs,2,2,cfg.multipathCorrelatorPositionChips,cfg);

Xall = [spoof.featuresAll; multi.featuresAll];
Xspecial = [spoof.featuresSpecial; multi.featuresSpecial];
y = [spoof.y; multi.y];
recordingId = [spoof.recordingId; multi.recordingId];
sourceTimeMs = [spoof.sourceTimeMs; multi.sourceTimeMs];
split = chronologicalSplit(y,recordingId,cfg.splitFractions);

trainMask = split==1;
testMask = split==3;

%% Stage 1: clean (0) vs interference (1/2)
stage1TrainLabel = double(y(trainMask)>0);
stage1Model = fitBinaryGaussianMembership(Xall(trainMask,:),stage1TrainLabel,[0 1]);
tInference = tic;
[~,stage1LogM] = predictBinaryGaussianMembership(stage1Model,Xall);

%% Stage 2: multipath (2) vs spoofing (1), trained only on abnormal epochs
stage2TrainMask = trainMask & y>0;
% Paper flowchart: output 0=multipath, 1=spoofing.
stage2TrainLabel = double(y(stage2TrainMask)==1);
stage2Model = fitBinaryGaussianMembership(Xspecial(stage2TrainMask,:),stage2TrainLabel,[0 1]);
[~,stage2LogM] = predictBinaryGaussianMembership(stage2Model,Xspecial);

epochPred = cascadeDecision(stage1LogM,stage2LogM);
epochScores = cascadeScores(stage1LogM,stage2LogM);
inferenceSeconds = toc(tInference);
epochMetrics = multiclassMetrics(y(testMask),epochPred(testMask),[0 1 2], ...
    epochScores(testMask,:));

%% Same 10 ms non-overlapping decision units as the proposed network
[stage1Win,meta,epochCells] = aggregateScores(stage1LogM,y,split,recordingId,sourceTimeMs,cfg.evalWindowMs);
stage2Win = zeros(height(meta),2);
for k=1:numel(epochCells)
    stage2Win(k,:)=mean(stage2LogM(epochCells{k},:),1,'omitnan');
end
windowPred = cascadeDecision(stage1Win,stage2Win);
windowTest = meta.split==3;
windowScores = cascadeScores(stage1Win,stage2Win);
windowMetrics = multiclassMetrics(meta.label(windowTest),windowPred(windowTest),[0 1 2], ...
    windowScores(windowTest,:));

featureSeconds = spoof.featureSeconds + multi.featureSeconds;
runtime = table({'DWT_feature_extraction';'Fuzzy_cascade_inference'}, ...
    [featureSeconds;inferenceSeconds],1e6*[featureSeconds/numel(y);inferenceSeconds/numel(y)], ...
    [0;16],{'epoch';'epoch'}, ...
    'VariableNames',{'Stage','TotalSeconds','MicrosecondsPerEpoch','LearnedScalarParameters','Unit'});

%% Save auditable results
epochTable = table(y,split,recordingId,sourceTimeMs,epochPred, ...
    stage1LogM(:,1),stage1LogM(:,2),stage2LogM(:,1),stage2LogM(:,2), ...
    epochScores(:,1),epochScores(:,2),epochScores(:,3), ...
    'VariableNames',{'label','split','recording_id','source_time_ms','prediction', ...
    'stage1_clean_log_membership','stage1_interference_log_membership', ...
    'stage2_multipath_log_membership','stage2_spoof_log_membership', ...
    'clean_score','spoof_score','multipath_score'});
scenarioPositions = [cfg.spoofCorrelatorPositionChips; ...
    cfg.multipathCorrelatorPositionChips];
epochTable.scenario_correlator_position_chips = scenarioPositions(epochTable.recording_id);

windowTable = meta;
windowTable.prediction = windowPred;
windowTable.stage1_clean_log_membership = stage1Win(:,1);
windowTable.stage1_interference_log_membership = stage1Win(:,2);
windowTable.stage2_multipath_log_membership = stage2Win(:,1);
windowTable.stage2_spoof_log_membership = stage2Win(:,2);
windowTable.clean_score = windowScores(:,1);
windowTable.spoof_score = windowScores(:,2);
windowTable.multipath_score = windowScores(:,3);
windowTable.scenario_correlator_position_chips = scenarioPositions(windowTable.recording_id);

writetable(epochMetrics.perClass,fullfile(cfg.outputDir,'fuzzy_epoch_per_class.csv'));
writetable(epochMetrics.summary,fullfile(cfg.outputDir,'fuzzy_epoch_summary.csv'));
writetable(windowMetrics.perClass,fullfile(cfg.outputDir,'fuzzy_10ms_per_class.csv'));
writetable(windowMetrics.summary,fullfile(cfg.outputDir,'fuzzy_10ms_summary.csv'));
writetable(runtime,fullfile(cfg.outputDir,'fuzzy_runtime.csv'));
writetable(epochTable,fullfile(cfg.outputDir,'fuzzy_epoch_predictions.csv'));
writetable(windowTable,fullfile(cfg.outputDir,'fuzzy_10ms_predictions.csv'));

results = struct();
results.config = cfg;
results.stage1Model = stage1Model;
results.stage2Model = stage2Model;
results.epochMetrics = epochMetrics;
results.windowMetrics = windowMetrics;
results.runtime = runtime;
results.epochPredictions = epochTable;
results.windowPredictions = windowTable;
results.waveletCoefficientLength = spoof.coefficientLength;
results.specialCoefficientIndices = cfg.specialCoefficientIndices;
save(fullfile(cfg.outputDir,'fuzzy_dwt_threeclass_results.mat'),'results','-v7.3');

if cfg.saveFigures
    saveFigures(results,Xall,Xspecial,testMask,windowTest);
end

fprintf('\nSingle-epoch test summary:\n'); disp(epochMetrics.summary);
fprintf('10-ms-window test summary:\n'); disp(windowMetrics.summary);
fprintf('Results saved to: %s\n',cfg.outputDir);
end

%% --------------------------- configuration ---------------------------
function cfg=defaultConfig()
rootDir=fileparts(mfilename('fullpath'));
cfg=struct();

%% 静态场景参数配置
% cfg.spoofFile='dataset/raw/asy_tim_0.1v_81corr.mat';
% cfg.multipathFile='dataset/raw/multipath_static_1_corr81.mat';
% cfg.spoofChannel=7;
% cfg.multipathChannel=3;
% cfg.spoofOnsetMs = 89500;     %-1.0chips channel7
% % cfg.spoofOnsetMs = 104000;     %-0.5chips channel7
% cfg.multipathOnsetMs = 83500;  %-1.7
% cfg.spoofCorrelatorPositionChips=-1.0;
% cfg.multipathCorrelatorPositionChips=-0.7; %5 -1.7chips  %3 -0.7chips
% cfg.useFullRecording=false; % true: load, evaluate and plot every available epoch
% cfg.preEventMs=10000;
% cfg.postEventMs=10000;
% cfg.outputDir=fullfile(rootDir,'results_fuzzy_static');


%% 动态场景参数配置
cfg.spoofFile='dataset/raw/asy_tim_0.5v_81corr.mat';
cfg.multipathFile='dataset/raw/multipath_dy_15m_corr81.mat';
cfg.spoofChannel=7;
cfg.multipathChannel=3;
cfg.spoofOnsetMs = 77700;
cfg.multipathOnsetMs = 79500;
cfg.spoofCorrelatorPositionChips=-1.0;
cfg.multipathCorrelatorPositionChips=-1.5; %5 -1.7chips  %3 -0.7chips
cfg.useFullRecording=false; % true: load, evaluate and plot every available epoch
cfg.preEventMs=10000;
cfg.postEventMs=10000;
cfg.outputDir=fullfile(rootDir,'results_fuzzy_dynamic');


%%
cfg.numCorrelators=81;
cfg.waveletName='sym4';
cfg.decompositionLevel=4;
% cfg.specialCoefficientIndices=[18 21 27 38 39 40];
cfg.specialCoefficientIndices=[17 18 20 21 27 38 39 40];   %70
% cfg.specialCoefficientIndices=[17 ];
cfg.smoothingHalfWindow=5;
cfg.smoothingMode='causal'; % 'causal' recommended; 'centered' follows paper Eq. 9 literally
cfg.normalizeProfile=false; % false is closer to the paper; true isolates shape from amplitude
cfg.evalWindowMs=10;
cfg.splitFractions=[0.60 0.20 0.20];
cfg.randomSeed=42;
cfg.saveFigures=true;
cfg.figureVisible='on';
cfg.closeFiguresAfterSave=false;
end

function cfg=mergeConfig(cfg,userCfg)
names=fieldnames(userCfg);
for i=1:numel(names)
    if ~isfield(cfg,names{i}), error('Unknown configuration field: %s',names{i}); end
    cfg.(names{i})=userCfg.(names{i});
end
end

function validateConfig(cfg)
assert(isfile(cfg.spoofFile),'Spoofing file not found: %s',cfg.spoofFile);
assert(isfile(cfg.multipathFile),'Multipath file not found: %s',cfg.multipathFile);
assert(cfg.numCorrelators==81,'The paper-aligned reproduction expects 81 correlators.');
assert(abs(sum(cfg.splitFractions)-1)<1e-12 && all(cfg.splitFractions>0), ...
    'splitFractions must be positive and sum to one.');
assert(any(strcmpi(cfg.smoothingMode,{'causal','centered'})), ...
    'smoothingMode must be causal or centered.');
assert(all(cfg.specialCoefficientIndices>=1),'Invalid DWT coefficient index.');
assert(isscalar(cfg.spoofCorrelatorPositionChips) && ...
    isfinite(cfg.spoofCorrelatorPositionChips), ...
    'spoofCorrelatorPositionChips must be a finite scalar.');
assert(isscalar(cfg.multipathCorrelatorPositionChips) && ...
    isfinite(cfg.multipathCorrelatorPositionChips), ...
    'multipathCorrelatorPositionChips must be a finite scalar.');
assert(islogical(cfg.useFullRecording) && isscalar(cfg.useFullRecording), ...
    'useFullRecording must be a logical scalar.');
assert(any(strcmpi(cfg.figureVisible,{'on','off'})), ...
    'figureVisible must be on or off.');
assert(islogical(cfg.closeFiguresAfterSave) && isscalar(cfg.closeFiguresAfterSave), ...
    'closeFiguresAfterSave must be a logical scalar.');
end

%% ----------------------------- data I/O -------------------------------
function out=loadAndExtractScenario(fileName,channelIdx,onsetMs,preMs,postMs, ...
    abnormalLabel,recordingId,correlatorPositionChips,cfg)
s=load(fileName,'trackData');
ch=s.trackData.beib1.channel(channelIdx);
lastAvailable=numel(ch.I_P);
if cfg.useFullRecording
    firstIdx=1;
    lastIdx=lastAvailable;
else
    firstIdx=max(1,onsetMs-preMs+1);
    lastIdx=min(lastAvailable,onsetMs+postMs);
end
assert(onsetMs>=firstIdx && onsetMs<lastIdx, ...
    'Event onset %d ms is outside the selected interval [%d, %d].', ...
    onsetMs,firstIdx,lastIdx);
idx=(firstIdx:lastIdx).';
[U,~]=extractCorrelatorIQ(ch,idx,cfg.numCorrelators);
profile=abs(U).';
clear U s

if cfg.normalizeProfile
    prompt=profile(:,(cfg.numCorrelators+1)/2);
    floorValue=max(eps,median(prompt,'omitnan')*1e-6);
    profile=profile./max(prompt,floorValue);
end

tFeature=tic;
[featuresAll,featuresSpecial,nCoeff]=dwtFeatures(profile,cfg);
featureSeconds=toc(tFeature);
y=zeros(numel(idx),1); y(idx>onsetMs)=abnormalLabel;
out=struct('featuresAll',featuresAll,'featuresSpecial',featuresSpecial, ...
    'coefficientLength',nCoeff,'y',y,'recordingId',repmat(recordingId,numel(idx),1), ...
    'sourceTimeMs',idx,'featureSeconds',featureSeconds, ...
    'correlatorPositionChips',correlatorPositionChips);
end

function [U,offsets]=extractCorrelatorIQ(ch,idx,nCorr)
assert(nCorr==81);
earlyI=cell(1,40); earlyQ=cell(1,40); lateI=cell(1,40); lateQ=cell(1,40);
for j=1:40
    ke=40-j;
    if ke==0
        earlyI{j}='I_E'; earlyQ{j}='Q_E';
    else
        earlyI{j}=sprintf('I_E_%d',ke); earlyQ{j}=sprintf('Q_E_%d',ke);
    end
    kl=j-1;
    if kl==0
        lateI{j}='I_L'; lateQ{j}='Q_L';
    else
        lateI{j}=sprintf('I_L_%d',kl); lateQ{j}=sprintf('Q_L_%d',kl);
    end
end
allI=[earlyI,{'I_P'},lateI]; allQ=[earlyQ,{'Q_P'},lateQ];
U=complex(zeros(nCorr,numel(idx)));
for k=1:nCorr
    assert(isfield(ch,allI{k}) && isfield(ch,allQ{k}), ...
        'Missing field %s or %s.',allI{k},allQ{k});
    iv=ch.(allI{k})(idx); qv=ch.(allQ{k})(idx);
    U(k,:)=double(iv(:).')+1i*double(qv(:).');
end
if isfield(ch,'corrFingers') && numel(ch.corrFingers)==nCorr
    offsets=double(ch.corrFingers(:));
    if any(diff(offsets)<=0), offsets=linspace(-2,2,nCorr).'; end
else
    offsets=linspace(-2,2,nCorr).';
end
end

%% -------------------------- feature extraction ------------------------
function [featuresAll,featuresSpecial,nCoeff]=dwtFeatures(profile,cfg)
n=size(profile,1);
[c0,~]=wavedec(profile(1,:),cfg.decompositionLevel,cfg.waveletName);
nCoeff=numel(c0);
assert(max(cfg.specialCoefficientIndices)<=nCoeff, ...
    'Requested special coefficient %d, but wavedec returns only %d coefficients.', ...
    max(cfg.specialCoefficientIndices),nCoeff);
C=zeros(n,nCoeff,'single');
C(1,:)=single(c0);
for i=2:n
    C(i,:)=single(wavedec(profile(i,:),cfg.decompositionLevel,cfg.waveletName));
end

m=cfg.smoothingHalfWindow;
if strcmpi(cfg.smoothingMode,'causal')
    Cs=movmean(C,[m 0],1,'omitnan');
else
    Cs=movmean(C,[m m],1,'omitnan');
end
featuresAll=[mean(Cs,2,'omitnan'),var(Cs,0,2,'omitnan')];
special=Cs(:,cfg.specialCoefficientIndices);
featuresSpecial=[mean(special,2,'omitnan'),var(special,0,2,'omitnan')];
featuresAll=double(featuresAll); featuresSpecial=double(featuresSpecial);
end

function split=chronologicalSplit(y,recordingId,fractions)
split=zeros(size(y),'uint8');
for r=unique(recordingId(:)).'
    for c=unique(y(recordingId==r)).'
        idx=find(recordingId==r & y==c); n=numel(idx);
        nTrain=max(1,floor(fractions(1)*n)); nVal=max(1,floor(fractions(2)*n));
        if nTrain+nVal>=n, nTrain=max(1,n-2); nVal=1; end
        split(idx(1:nTrain))=1;
        split(idx(nTrain+1:nTrain+nVal))=2;
        split(idx(nTrain+nVal+1:end))=3;
    end
end
assert(all(split>0));
end

%% -------------------------- fuzzy classifiers -------------------------
% function model=fitBinaryGaussianMembership(X,y,classes)
% model.classes=classes(:).'; model.mu=zeros(2,size(X,2)); model.sigma=model.mu;
% for i=1:2
%     Xi=X(y==classes(i),:);
%     assert(~isempty(Xi),'Training class %d is empty.',classes(i));
%     model.mu(i,:)=mean(Xi,1,'omitnan');
%     model.sigma(i,:)=std(Xi,0,1,'omitnan');
% end
% positive=model.sigma(model.sigma>0 & isfinite(model.sigma));
% if isempty(positive), sigmaFloor=1e-9; else, sigmaFloor=max(median(positive)*1e-3,1e-9); end
% model.sigma=max(model.sigma,sigmaFloor);
% end
%
% function [pred,logM]=predictBinaryGaussianMembership(model,X)
% logM=zeros(size(X,1),2);
% for i=1:2
%     z=(X-model.mu(i,:))./model.sigma(i,:);
%     % Gaussian fuzzy membership mu=exp(-z^2/2); product is summed in log-space.
%     logM(:,i)=-0.5*sum(z.^2,2);
% end
% [~,ix]=max(logM,[],2); pred=model.classes(ix).';
% end

function model=fitBinaryGaussianMembership(X,y,classes)
% 多元高斯隶属函数：保存mu、完整协方差Sigma，正则化防止奇异
model.classes = classes(:).';
nFeat = size(X,2);
model.mu = zeros(2, nFeat);
model.Sigma = cell(2,1);   % 每一类的协方差矩阵
model.regLam = 1e-6;      % 协方差正则系数，可微调

for i = 1:2
    Xi = X(y == classes(i), :);
    assert(~isempty(Xi),'Training class %d is empty.',classes(i));
    mu_i = mean(Xi,1,'omitnan');
    model.mu(i,:) = mu_i;

    % 去除含有NaN的样本行，计算协方差
    validMask = ~any(isnan(Xi),2);
    XiValid = Xi(validMask,:);
    if size(XiValid,1) < nFeat
        warning('Class %d valid samples less than feature dimension, heavy regularization apply',classes(i));
    end
    SigmaRaw = cov(XiValid,0);
    % 正则：给对角线加小值，保证正定可逆
    SigmaReg = SigmaRaw + model.regLam * eye(nFeat);
    model.Sigma{i} = SigmaReg;
end
end

function [pred,logM]=predictBinaryGaussianMembership(model,X)
N = size(X,1);
logM = zeros(N,2);

for i = 1:2
    mu_i = model.mu(i,:);
    Sigma_i = model.Sigma{i};

    % 预处理NaN：简单填充为类别均值
    Xcur = X;
    nanRows = any(isnan(Xcur),2);
    Xcur(nanRows,:) = repmat(mu_i,sum(nanRows),1);

    % (x‑mu)
    dx = Xcur - repmat(mu_i,N,1);

    % 多元高斯 log pdf
    % log|Sigma|
    logDetSigma = log(det(Sigma_i));
    % (x‑mu) Σ⁻¹ (x‑mu)^T 马氏距离平方
    invSigma = inv(Sigma_i);
    mahalSq = sum( (dx * invSigma) .* dx , 2 );

    logM(:,i) = -0.5*( logDetSigma + mahalSq );
end

[~,ix] = max(logM,[],2);
pred = model.classes(ix).';
end


function pred=cascadeDecision(stage1LogM,stage2LogM)
pred=zeros(size(stage1LogM,1),1);
interference=stage1LogM(:,2)>stage1LogM(:,1);
spoof=stage2LogM(:,2)>stage2LogM(:,1);
pred(interference & spoof)=1;
pred(interference & ~spoof)=2;
end

function scores=cascadeScores(stage1LogM,stage2LogM)
% Convert the two binary fuzzy decisions into comparable three-class scores.
% The product form preserves the cascade probability semantics:
% clean = P(clean), spoof = P(interference)P(spoof|interference),
% multipath = P(interference)P(multipath|interference).
p1=softmaxRows(stage1LogM);
p2=softmaxRows(stage2LogM);
scores=[p1(:,1),p1(:,2).*p2(:,2),p1(:,2).*p2(:,1)];
scores=scores./max(sum(scores,2),eps);
end

function p=softmaxRows(x)
x=x-max(x,[],2);
e=exp(x);
p=e./max(sum(e,2),eps);
end

function [scoreW,meta,epochCells]=aggregateScores(scores,y,split,recordingId,timeMs,W)
scoreW=zeros(0,size(scores,2)); labelW=zeros(0,1); splitW=zeros(0,1,'uint8');
recordW=zeros(0,1); startW=zeros(0,1); endW=zeros(0,1); epochCells=cell(0,1);
for r=unique(recordingId(:)).'
    for s=uint8(1:3)
        for c=unique(y(recordingId==r & split==s)).'
            idx=find(recordingId==r & split==s & y==c); nFull=floor(numel(idx)/W);
            for k=1:nFull
                block=idx((k-1)*W+1:k*W);
                scoreW(end+1,:)=mean(scores(block,:),1,'omitnan'); %#ok<AGROW>
                labelW(end+1,1)=c; splitW(end+1,1)=s; recordW(end+1,1)=r; %#ok<AGROW>
                startW(end+1,1)=timeMs(block(1)); endW(end+1,1)=timeMs(block(end)); %#ok<AGROW>
                epochCells{end+1,1}=block; %#ok<AGROW>
            end
        end
    end
end
meta=table(labelW,splitW,recordW,startW,endW, ...
    'VariableNames',{'label','split','recording_id','start_ms','end_ms'});
end

%% ------------------------------ metrics -------------------------------
function metrics=multiclassMetrics(y,pred,classes,scores)
% Standard three-class metrics.  scores(:,i) is the continuous score for
% classes(i), used for one-vs-rest AUC.  AUC is computed locally so this
% function does not require perfcurve/Statistics Toolbox.
if nargin<4 || isempty(scores)
    scores=double(pred(:)==classes(:).');
end
C=zeros(numel(classes),numel(classes));
for i=1:numel(classes)
    for j=1:numel(classes)
        C(i,j)=sum(y==classes(i) & pred==classes(j));
    end
end
precision=zeros(numel(classes),1); recall=precision; f1=precision;
specificity=precision; auc=precision; support=sum(C,2);
for i=1:numel(classes)
    precision(i)=C(i,i)/max(sum(C(:,i)),1);
    recall(i)=C(i,i)/max(support(i),1);
    f1(i)=2*precision(i)*recall(i)/max(precision(i)+recall(i),eps);
    negative=sum(C(:))-support(i);
    tn=sum(C(:))-support(i)-sum(C(:,i))+C(i,i);
    specificity(i)=tn/max(negative,1);
    auc(i)=binaryAuc(y==classes(i),scores(:,i));
end
weightedF1=sum(f1.*support)/max(sum(support),1);
metrics.perClass=table(classes(:),precision,recall,specificity,f1,auc,support, ...
    'VariableNames',{'Class','Precision','Recall','Specificity','F1','AUC','Support'});
metrics.summary=table(sum(diag(C))/max(sum(C,'all'),1),mean(f1),weightedF1, ...
    mean(recall),mean(auc),mean(specificity), ...
    'VariableNames',{'Accuracy','MacroF1','WeightedF1','MacroRecall', ...
    'MacroAUC','MacroSpecificity'});
metrics.confusion=C;
end

function auc=binaryAuc(positive,score)
positive=logical(positive(:)); score=double(score(:));
valid=isfinite(score);
positive=positive(valid); score=score(valid);
nPos=sum(positive); nNeg=sum(~positive);
if nPos==0 || nNeg==0
    auc=NaN;
    return;
end
[~,order]=sort(score,'ascend');
sorted=score(order);
sortedPos=positive(order);
ranks=zeros(size(sorted)); k=1; n=numel(sorted);
while k<=n
    j=k;
    while j<n && sorted(j+1)==sorted(k), j=j+1; end
    ranks(k:j)=(k+j)/2;
    k=j+1;
end
rankSum=sum(ranks(sortedPos));
auc=(rankSum-nPos*(nPos+1)/2)/(nPos*nNeg);
end

%% ------------------------------- plots --------------------------------
function saveFigures(results,Xall,Xspecial,testMask,windowTest)
outDir=results.config.outputDir;
cfg=results.config;
f=figure('Visible',cfg.figureVisible,'Color','w','Position',[100 100 1000 420]);
colors=lines(3);
subplot(1,2,1); hold on;
for c=0:2
    scatter(Xall(testMask & results.epochPredictions.label==c,1), ...
        Xall(testMask & results.epochPredictions.label==c,2),8,colors(c+1,:),'.');
end
xlabel('Mean of all DWT coefficients'); ylabel('Variance'); title('Stage 1 features'); grid on;
legend({'Clean','Spoof','Multipath'},'Location','best');
subplot(1,2,2); hold on;
% Stage 2 is trained and interpreted only on abnormal samples.  Clean
% samples must therefore not appear in this feature-space panel.
for c=1:2
    scatter(Xspecial(testMask & results.epochPredictions.label==c,1), ...
        Xspecial(testMask & results.epochPredictions.label==c,2),8,colors(c+1,:),'.');
end
xlabel('Mean of selected coefficients'); ylabel('Variance'); title('Stage 2 features'); grid on;
legend({'Spoof','Multipath'},'Location','best');
sgtitle('DWT feature space on the held-out test set');
exportFigurePair(f,fullfile(outDir,'fuzzy_dwt_feature_space'));
closeFigureIfRequested(f,cfg);

C=results.windowMetrics.confusion;
f=figure('Visible',cfg.figureVisible,'Color','w'); imagesc(C); axis image; colorbar; colormap(parula);
xticks(1:3); yticks(1:3); xticklabels({'Clean','Spoof','Multipath'});
yticklabels({'Clean','Spoof','Multipath'}); xlabel('Predicted'); ylabel('True');
title(sprintf('DWT-fuzzy, %d ms windows',results.config.evalWindowMs));
for i=1:3, for j=1:3, text(j,i,num2str(C(i,j)),'HorizontalAlignment','center','Color','w'); end, end
exportFigurePair(f,fullfile(outDir,'fuzzy_10ms_confusion'));
closeFigureIfRequested(f,cfg);

% Visualize the final fuzzy detection quantities on each complete recording.
% Stage-1 margin > 0 means interference; stage-2 margin > 0 means spoofing
% rather than multipath. The last panel shows the cascaded final decision.
P=results.windowPredictions;
recordNames={'Spoofing recording','Multipath recording'};
fileNames={'spoofing','multipath'};
scenarioPositions=[cfg.spoofCorrelatorPositionChips,cfg.multipathCorrelatorPositionChips];
if cfg.useFullRecording, timeScope='full time span'; else, timeScope='selected time interval'; end
scoreColor=[0.16 0.35 0.58];
thresholdColor=[0.78 0.20 0.20];
truthColor=[0.10 0.10 0.10];
predColor=[0.17 0.55 0.42];

for r=unique(P.recording_id(:)).'
    idx=find(P.recording_id==r);
    [~,order]=sort(P.start_ms(idx)); idx=idx(order);
    tSec=0.5*(P.start_ms(idx)+P.end_ms(idx))/1000;
    trueLabel=P.label(idx);
    prediction=P.prediction(idx);
    stage1Margin=P.stage1_interference_log_membership(idx)-P.stage1_clean_log_membership(idx);
    stage2Margin=P.stage2_spoof_log_membership(idx)-P.stage2_multipath_log_membership(idx);
    eventIdx=find(trueLabel~=0,1,'first');
    if isempty(eventIdx), eventTime=NaN; else, eventTime=tSec(eventIdx); end

    f=figure('Visible',cfg.figureVisible,'Color','w','Position',[80 80 1200 720]);
    tl=tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

    ax1=nexttile(tl); hold(ax1,'on');
    plot(ax1,tSec,stage1Margin,'Color',scoreColor,'LineWidth',0.9, ...
        'DisplayName','Interference - clean log-membership');
    yline(ax1,0,'--','Color',thresholdColor,'LineWidth',1.2,'DisplayName','Decision boundary');
    if isfinite(eventTime)
        xline(ax1,eventTime,':','Color',[0.12 0.12 0.12],'LineWidth',1.0,'DisplayName','Event onset');
        addEventShade(ax1,eventTime,max(tSec),trueLabel(eventIdx));
    end
    ylabel(ax1,'Stage-1 margin'); title(ax1,'Stage 1: clean versus interference','FontWeight','normal');
    grid(ax1,'on'); box(ax1,'on'); legend(ax1,'Location','best','Box','off');

    ax2=nexttile(tl); hold(ax2,'on');
    plot(ax2,tSec,stage2Margin,'Color',scoreColor,'LineWidth',0.9);
    yline(ax2,0,'--','Color',thresholdColor,'LineWidth',1.2);
    if isfinite(eventTime)
        xline(ax2,eventTime,':','Color',[0.12 0.12 0.12],'LineWidth',1.0);
        addEventShade(ax2,eventTime,max(tSec),trueLabel(eventIdx));
    end
    ylabel(ax2,'Stage-2 margin');
    title(ax2,'Stage 2: spoofing (>0) versus multipath (<0)','FontWeight','normal');
    grid(ax2,'on'); box(ax2,'on');

    ax3=nexttile(tl); hold(ax3,'on');
    stairs(ax3,tSec,trueLabel,'Color',truthColor,'LineWidth',1.5,'DisplayName','True class');
    stairs(ax3,tSec,prediction,'Color',predColor,'LineWidth',1.0,'DisplayName','Predicted class');
    if isfinite(eventTime)
        xline(ax3,eventTime,':','Color',[0.12 0.12 0.12],'LineWidth',1.0,'HandleVisibility','off');
    end
    ylim(ax3,[-0.25 2.25]); yticks(ax3,[0 1 2]);
    yticklabels(ax3,{'Clean','Spoof','Multipath'});
    xlabel(ax3,'Source time (s)'); ylabel(ax3,'Class');
    title(ax3,'Final cascaded three-class decision','FontWeight','normal');
    grid(ax3,'on'); box(ax3,'on'); legend(ax3,'Location','best','Box','off');

    if r<=numel(recordNames), recTitle=recordNames{r}; else, recTitle=sprintf('Recording %d',r); end
    if r<=numel(scenarioPositions), positionChips=scenarioPositions(r); else, positionChips=NaN; end
    title(tl,sprintf(['%s (correlator position %.1f chips): DWT-fuzzy ' ...
        'statistics over the %s'],recTitle,positionChips,timeScope),'FontWeight','bold');
    if r<=numel(fileNames), suffix=fileNames{r}; else, suffix=sprintf('recording_%d',r); end
    exportFigurePair(f,fullfile(outDir,['fuzzy_detection_statistics_' suffix]));
    closeFigureIfRequested(f,cfg);
end
end

function addEventShade(ax,xStart,xEnd,eventLabel)
yl=ylim(ax);
if eventLabel==1
    shadeColor=[0.94 0.55 0.48];
else
    shadeColor=[0.95 0.72 0.35];
end
h=patch(ax,[xStart xEnd xEnd xStart],[yl(1) yl(1) yl(2) yl(2)],shadeColor, ...
    'FaceAlpha',0.10,'EdgeColor','none','HandleVisibility','off');
uistack(h,'bottom');
ylim(ax,yl);
end

function exportFigurePair(f,baseName)
exportgraphics(f,[baseName '.png'],'Resolution',300);
exportgraphics(f,[baseName '.pdf'],'ContentType','vector');
end

function closeFigureIfRequested(f,cfg)
if cfg.closeFiguresAfterSave
    close(f);
end
end

function plot_prediction_timelines(runDirs,recordIDs,outputDir)
%PLOT_PREDICTION_TIMELINES Plot raw predictions from the Python trainer.
% Example: plot_prediction_timelines({runA,runB},{'bs6','bs7'},'outputs/figures')
% One column per record, one row per model. No smoothing/downsampling.
% Percentages are accuracy within each physical-state interval, not recall.
    models={'cnn','lstm','resnet1d','transformer','proposed'};
    labels={'CNN','LSTM','ResNet1D','Transformer','Proposed'};
    colors=[.83 .37 .14;.45 .31 .61;.25 .55 .45;.65 .46 .25;0 .447 .741];
    assert(numel(runDirs)==numel(recordIDs),'One run directory per record.');
    if ~isfolder(outputDir),mkdir(outputDir);end
    f=figure('Color','w','Position',[100 100 1200 850],'Visible','off');
    tiledlayout(5,numel(runDirs),'TileSpacing','compact','Padding','compact');
    allStats=table();
    for m=1:5
        for p=1:numel(runDirs)
            file=fullfile(runDirs{p},models{m},['target_test_' recordIDs{p} '.csv']);
            T=readtable(file);T=sortrows(T,'start_ms');
            if m==1
                reference{p}=T(:,{'start_ms','end_ms','label','physical_state'}); %#ok<AGROW>
            else
                assert(isequal(reference{p},T(:,{'start_ms','end_ms','label','physical_state'})), ...
                    'Models must use identical test windows.');
            end
            nexttile;hold on
            stairs(T.start_ms/1000,T.label,'k--','LineWidth',1.1);
            plot((T.start_ms+T.end_ms)/2000,T.prediction,'.', ...
                'Color',colors(m,:),'MarkerSize',2);
            edges=[1;find(diff(T.physical_state)~=0)+1;height(T)+1];
            for j=1:numel(edges)-1
                ix=edges(j):edges(j+1)-1;
                accuracy=mean(T.prediction(ix)==T.label(ix));
                left=T.start_ms(ix(1))/1000;right=T.end_ms(ix(end))/1000;
                text((left+right)/2,2.38,sprintf('%.2f%%',100*accuracy), ...
                    'HorizontalAlignment','center','FontSize',9);
                if j>1,xline(left,':','Color',[.6 .6 .6]);end
                allStats=[allStats;table(string(models{m}),string(recordIDs{p}), ...
                    T.physical_state(ix(1)),left,right,numel(ix),accuracy, ...
                    'VariableNames',{'model','record','state','start_s','end_s','samples','accuracy'})]; %#ok<AGROW>
            end
            ylim([-.2 2.7]);xlim([T.start_ms(1) T.end_ms(end)]/1000);yticks(0:2);
            set(gca,'FontName','Arial','FontSize',9,'Box','off','TickDir','out');
            ylabel(labels{m});
            if m==1,title(strrep(recordIDs{p},'_',' '));end
            if m==5,xlabel('Time from first selected epoch (s)');end
        end
    end
    savefig(f,fullfile(outputDir,'prediction_timelines.fig'));
    exportgraphics(f,fullfile(outputDir,'prediction_timelines.pdf'),'ContentType','vector');
    exportgraphics(f,fullfile(outputDir,'prediction_timelines.png'),'Resolution',300);
    writetable(allStats,fullfile(outputDir,'stage_accuracy.csv'));
    close(f);
end

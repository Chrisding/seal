function plot_pr(scoreDir, methodName, numCls, plotDir, objectNames, flagAP)
if(exist(plotDir, 'file')==0)
    mkdir(plotDir);
end
numMtd = length(scoreDir);
assert(length(scoreDir)==length(methodName), ...
    'Number of input dirs must be equal to input method names!');

for idxCls = 1:numCls
    fprintf(['Plotting class %d "' objectNames{idxCls} '" precision-recall curve\n'], idxCls);
    resultCatLst = cell(numMtd, 1);

    for idxMtd = 1:numMtd
        s = load(fullfile(scoreDir{idxMtd}, ['/class_' num2str(idxCls) '.mat']));
        names = fieldnames(s);
        resultCatLst{idxMtd} = s.(names{1});
    end
    
    % plot figure
    [F_ODS, ~, AP, H] = plot_eval_multiple(resultCatLst);
    legendLst = cell(numMtd, 1);
    for idxMtd = 1:numMtd
        if(flagAP)
            legendLst{idxMtd, 1} = ['[F=' num2str(F_ODS(idxMtd), '%1.3f') ' AP=' num2str(AP(idxMtd), '%1.3f') '] ' methodName{idxMtd, 1}];
        else
            legendLst{idxMtd, 1} = ['[F=' num2str(F_ODS(idxMtd), '%1.3f') '] ' methodName{idxMtd, 1}];
        end
    end
    set(gca,'fontsize',10)
    title(objectNames{idxCls},'FontSize',14,'FontWeight','bold')
    xlabel('Recall','FontSize',14,'FontWeight','bold')
    ylabel('Precision','FontSize',14,'FontWeight','bold')
    hLegend = legend(H, legendLst, 'Location', 'SouthWest');
    set(hLegend, 'FontSize', 12);
    
    % save figure
    print(gcf, fullfile(plotDir, ['/class_' num2str(idxCls, '%03d') '.pdf']),'-dpdf')
    close(gcf);
end

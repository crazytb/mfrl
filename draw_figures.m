clearvars

raalgos = ["RA2C", "recurrent", "vanilla", "fixedprob"];

for raalgo = raalgos
    ra2c = readmatrix(strcat('test_log_', raalgo, '_20240820_174021.csv'), 'NumHeaderLines', 1);
    recurrent = readmatrix(strcat('test_log_', raalgo, '_20240820_174021.csv'), 'NumHeaderLines', 1);
    vanilla = readmatrix(strcat('test_log_', raalgo, '_20240820_174021.csv'), 'NumHeaderLines', 1);
    fixedprob = readmatrix(strcat('test_log_', raalgo, '_20240820_174021.csv'), 'NumHeaderLines', 1);
end



% 
%     %% AoI distribution figure
%     aois_deepaaqm = reshape(ra2c(:, 7:end), [], 1);
%     aois_codel = reshape(recurrent(:, 7:end), [], 1);
%     aois_sred = reshape(vanilla(:, 7:end), [], 1);
% 
%     [counts_deepaaqm, edges_deepaaqm] = histcounts(aois_deepaaqm, 100);
%     [counts_codel, edges_codel] = histcounts(aois_codel, 100);
%     [counts_sred, edges_sred] = histcounts(aois_sred, 100);
% 
%     cdf_deepaaqm = cumsum(counts_deepaaqm)/sum(counts_deepaaqm);
%     cdf_codel = cumsum(counts_codel)/sum(counts_codel);
%     cdf_sred = cumsum(counts_sred)/sum(counts_sred);
% 
%     x = 100*linspace(0, 1);
% 
%     figure()
%     hold on;
%     plot(x, cdf_deepaaqm, 'LineWidth', 1)
%     plot(x, cdf_codel, 'LineWidth', 1)
%     plot(x, cdf_sred, 'LineWidth', 1)
%     xline(20, 'LineWidth', 1)
%     legend('DeepAAQM', 'CoDel', 'SRED')
%     title("Radio access algorithm: " + raalgo + ", " + "Number of nodes: " + num2str(numnode));
%     xlabel('Peak AoI (ms)')
%     ylabel('Frequency')
%     grid on;
% 
%     %% AoI evolution figure
%     iteration = 0;
%     startfrom = (333*iteration)+1;
%     endto = 333*(iteration+1);
% 
%     figure()
%     hold on;
%     plot(ra2c(startfrom:endto, 7), 'LineWidth', 1, 'LineStyle', '-')
%     plot(recurrent(startfrom:endto, 7), 'LineWidth', 1, 'LineStyle', '--')
%     plot(vanilla(startfrom:endto, 7), 'LineWidth', 1, 'LineStyle', '-.')
%     yline(0.2, 'LineWidth', 1)
%     legend('DeepAAQM', 'CoDel', 'SRED', 'Location', 'best')
%     title("Radio access algorithm: " + raalgo + ", " + "Number of nodes: " + num2str(numnode));
%     xlabel('Decision epochs')
%     ylabel('Peak AoI (ms)')        
% 
%     %% Print out some results
%     total_consumed_energy_deepaaqm = ra2c(333:333:end, 6)/100000;
%     total_consumed_energy_codel = recurrent(333:333:end, 6)/100000;
%     total_consumed_energy_sred = vanilla(333:333:end, 6)/100000;
% 
%     % In milliwatts
%     fprintf('=========================\n');
%     fprintf('raalgo: %s\n', raalgo);
%     fprintf('numnode: %d\n', numnode);
%     fprintf('total_consumed_energy_deepaaqm: %.2f mW\n', mean(total_consumed_energy_deepaaqm));
%     fprintf('total_consumed_energy_codel: %.2f mW\n', mean(total_consumed_energy_codel));
%     fprintf('total_consumed_energy_sred: %.2f mW\n', mean(total_consumed_energy_sred));
% end
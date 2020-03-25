input = 'C:/WkDir/BSR/BSDS500/Results/Segmentation/';
output = 'C:/WkDir/BSR/BSDS500/Results/ResInMat/';
mkdir(output);

iids = dir(fullfile(input, '*'));
for i = 3:numel(iids)
    if iids(i).isdir
        continue;
    end
    fin = fopen(fullfile(input, iids(i).name));
    head = fread(fin, 5, '*char');
    if ~strcmp(num2str(head)', 'CmMat')
        disp 'Warning: The data does not match...';
    end
    matCnt = fread(fin, 1, 'int');
    segs = {};
    for j = 1:matCnt
       headData = fread(fin, 3, 'int');
       segs{end+1} = fread(fin, [headData(1), headData(2)], 'ushort');
       segs{end} = segs{end}';
    end
    fclose(fin);
    
    save([output num2str(str2num(iids(i).name)) '.mat'], 'segs');
end

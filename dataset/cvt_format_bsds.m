%% Convert origin bsds data set in matlab format to our C++ format
function cvt_frt_bsds()
clear; clc;

% the directory of origin bsds data set in matlab format
input_dir = 'C:/WkDir/BSR/SegBench/BSDS500/';

% the output directory of data set in C++ format
output_dir = 'C:/WkDir/BSR/BSDS500/';

set_names = {'test','train','val'};

bdry_anns_dir = fullfile(output_dir,'Annotations','Boundaries');
segs_anns_dir = fullfile(output_dir,'Annotations','Segmentation');
imgs_dir = fullfile(output_dir,'JPEGImages');
idxs_dir = fullfile(output_dir,'ImageSets','Main');
bdry_dir = fullfile(output_dir,'GroundTruth','Boundaries');
seg_dir = fullfile(output_dir,'GroundTruth','Segmentation');
res_dir = fullfile(output_dir,'Results');

mkdir(output_dir);
mkdir(bdry_anns_dir);
mkdir(segs_anns_dir);
mkdir(imgs_dir);
mkdir(idxs_dir);
mkdir(bdry_dir);
mkdir(seg_dir);
mkdir(res_dir);

for n = 1:length(set_names)
   in_imgs = fullfile(input_dir,'image',set_names{n});
   iids = dir(fullfile(in_imgs,'*.jpg'));
   fid = fopen(fullfile(idxs_dir,strcat(set_names{n},'.txt')),'w');
   for i = 1:length(iids)
      index = str2num(iids(i).name(1:end-4)); 
      fprintf(fid,'%06d\n',index);
      I = imread(fullfile(in_imgs,iids(i).name));
      imwrite(I,fullfile(imgs_dir,sprintf('%06d.jpg',index)));
   end
   fclose(fid);
   
   in_mats = fullfile(input_dir,'groundTruth',set_names{n});
   iids = dir(fullfile(in_mats,'*.mat'));
   for i = 1:length(iids)
       index = str2num(iids(i).name(1:end-4));
       load(fullfile(in_mats,iids(i).name));
       
       bdry_fid = fopen(fullfile(bdry_anns_dir,sprintf('%06d',index)),'w');
       fwrite(bdry_fid,'CmMat','*char');
       fwrite(bdry_fid,length(groundTruth),'int');
       for j = 1:length(groundTruth)
          s = size(groundTruth{j}.Boundaries);
          fwrite(bdry_fid,[s(2) s(1) 0],'int');
          fwrite(bdry_fid,groundTruth{j}.Boundaries','uint8');
          
          bdry = ~groundTruth{j}.Boundaries;
          imwrite(bdry, fullfile(bdry_dir,sprintf('%06d_%d.png',index,j)));
       end
       fclose(bdry_fid);
       
       segs_fid = fopen(fullfile(segs_anns_dir,sprintf('%06d',index)),'w');
       fwrite(segs_fid,'CmMat','*char');
       fwrite(segs_fid,length(groundTruth),'int');
       for j = 1:length(groundTruth)
          s = size(groundTruth{j}.Segmentation);
          fwrite(segs_fid,[s(2) s(1) 2],'int');
          fwrite(segs_fid,groundTruth{j}.Segmentation','uint16');
          
          seg = groundTruth{j}.Segmentation;
          color3u = label2rgb(seg);
          imwrite(color3u, fullfile(seg_dir,sprintf('%06d_%d.png',index,j)));
       end
       fclose(segs_fid);
   end
end

%% convert label to rgb image
function [color3u] = label2rgb(label)
s = size(label);
color3u = zeros(s(1),s(2),3);
color3u = uint8(color3u);
for r = 1:s(1)
   for c = 1:s(2)
      if c>1 && label(r,c-1)==label(r,c)
          color3u(r,c,:)=color3u(r,c-1,:);
      else
          lab = label(r,c);
          for i = 0:7
             color3u(r,c,1) = bitor(color3u(r,c,1),uint8(bitshift(bitand(bitshift(lab,-0),1),7-i)));
             color3u(r,c,2) = bitor(color3u(r,c,2),uint8(bitshift(bitand(bitshift(lab,-1),1),7-i)));
             color3u(r,c,3) = bitor(color3u(r,c,3),uint8(bitshift(bitand(bitshift(lab,-2),1),7-i)));
             lab = bitshift(lab,-3);
             if lab <= 0
                break; 
             end
          end
      end
   end
end




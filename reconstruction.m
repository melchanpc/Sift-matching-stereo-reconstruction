clear; clc; close all;

%% Task 1 : Load and join images

% Load the images
im1 = imread('im1.jpg');
im2 = imread('im2.jpg');
im3 = imread('im3.jpg');
im4 = imread('im4.jpg');

% Load the keypoints data
imKp1 =load('im1.sift');
imKp2 =load('im2.sift');
imKp3 =load('im3.sift');
imKp4 =load('im4.sift');

% Initialize parameters for Task 6 Part 3
wBox = 37;   % width of box
bReal = zeros(3,1);     % baseline

% Filter kernels
gaussFilt = (1/159)*[2 4 5 4 2; 4 9 12 9 4; 5 12 15 12 5; 4 9 12 9 4; 2 4 5 4 2];
sobFilt = [-1 0 1; -2 0 2; -1 0 1];

% Filter and get edges for the images - use image3 cuz its most centred
imLFilt = double(rgb2gray(im3));
imLFilt = conv2(imLFilt,gaussFilt,'same');
imLFilt = conv2(imLFilt,sobFilt,'same');  imLFilt = mat2gray(abs(imLFilt));
thr = graythresh(imLFilt);  imLFilt = imbinarize(imLFilt,thr);
% figure; imshow(imLFilt);
sumWhite = sum(imLFilt); sumWhite([1:2 end-1:end]) = 0;
[~, locs] = findpeaks(sumWhite);
wBoxImPx = locs(end) - locs(1);

% Repeat for each image pair
fprintf('The estimated baselines are: \n');
for ii = 2:4
    
    switch ii
        case 2
            imR = im2; imKpR = imKp2;
            
        case 3
            imR = im3; imKpR = imKp3;
            
        case 4
            imR = im4; imKpR = imKp4;
    end
            
    pairName = sprintf('Image1 - Image%d',ii);
    
    % Displays the joined image
    imJoin = [im1 imR];
    figure; imshow(imJoin);
    title(strcat(pairName,': Original images joined together'));
    
    
    %% Tasks 2 and 3: Read and show SIFT keypoint locations

    % Get feature points as a N*2 matrix
    featurePts = [imKp1(:,1:2); imKpR(:,1:2)+[size(im1,2) 0]];

    % Mark the joined image
    imMarkedJoin = insertMarker(imJoin,featurePts,'x','color','r','size',5);

    % Displays the marked joined image
    figure; imshow(imMarkedJoin);
    title(strcat(pairName,': Original images marked with keypoints'));

    %% Tasks 4 and 5: Match SIFT keypoints and show matches

    Nkp = length(imKp1);    % number of keypoints

    % index of right image that corresponds to the ith point from the left
    rIdx = zeros(Nkp,1);    

    % Loop for all keypoints on the left image
    for k = 1:Nkp

        % Euclidean distance
        dist = sqrt(sum((imKp1(k,5:end) - imKpR(:,5:end)).^2,2));
        [dist, idx] = sort(dist,'ascend');

        % Compare 2 smallest distances
        if (dist(1)/dist(2) < 0.5)
            rIdx(k) = idx(1);
        end

    end

    % Find number of matches in both images
    lIdx = find(rIdx>0);

    % Get the coordinates that have match in both images
    ptsMatL = imKp1(lIdx,1:2);
    ptsMatR = imKpR(rIdx(lIdx),1:2);

    % Mark the points and draw lines
    matchMark = insertMarker(imMarkedJoin,[ptsMatL; ptsMatR + [size(im1,2) 0] ],'x','color','g','size',5);
    figure; imshow(matchMark);  hold on;
    for k = 1:length(lIdx)
        line([ptsMatL(k,1)  ptsMatR(k,1)+ size(im1,2) ],[ptsMatL(k,2)  ptsMatR(k,2) ],'color','g');
    end
    hold off; title(strcat(pairName,': Matched points identified'));

    %% Task 6: Stereo reconstruction
    
    flen = 1; bline = 1;  % focal length and baseline
    disp = (  ptsMatR(:,1) - ptsMatL(:,1) );     % disparity
%     disp = disp.*fac;
    
    depth = flen*bline./disp;               % estimated depth (z value)
    
%     bline = [2.7248 6.4680 9.6321];
%     depth = flen*bline(ii-1)./disp; 
    
    mZ = mean(depth); sZ = std(depth);
    goodIdx = (depth < mZ + sZ) & (depth > mZ - sZ) ;
    
    z = mean(depth(goodIdx));
    D = mean(disp(goodIdx));
    
    x1 = depth.*[ptsMatL ones(length(ptsMatL),1)];  % 3d-coordinates of the points in image1's frame
    figure(4);  col = zeros(1,3); col(ii-1) = 1; % colour
    scatter3(x1(:,1),x1(:,2),x1(:,3),5,col,'filled','DisplayName',pairName');
    hold on;
    
    % Estimate baselines 
    bReal(ii-1) = wBox*D/wBoxImPx;
    fprintf('   '); fprintf(pairName); fprintf(': %.4f \n',bReal(ii-1));
    
end
figure(4);
legend('show','Location','best'); hold off; title('Reconstructed points for each image pair');
xlabel('x'); ylabel('y'); zlabel('z'); axis(gca,'tight'); %zlim([0 0.05]);

%%  Task 7: Reprojection of 3D points

disp1 = bReal(2)/bReal(3)*disp;
pts = ptsMatL + [disp1 zeros(length(ptsMatL),1)];
imRecon3 = insertMarker(im3,pts,'x','color','r','size',5);
im1Marked = insertMarker(im1,ptsMatL,'x','color','r','size',5);
figure;
subplot(1,2,1); imshow(im1Marked); title('Points from Image1');
subplot(1,2,2); imshow(imRecon3);  title('Reprojected points on Image3');

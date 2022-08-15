% DMD_GrayTraffic.m
% Tested on the following builds:  MATLAB R2020b
%
% (C) 2022 Joseph Blochberger (jblochb2@jhu.edu).  All rights reserved.
% Feel free to share and use this script with associated files based on the
% following creative commons license: Attribution-NonCommercial-ShareAlike
% 4.0 International (CC BY-NC-SA 4.0).  For more information, see
% creativecommons.org/licenses/by-na-sa/4.0/
%
% Kindly cite as
%       Blochberger, J. 2021. Dynamic Mode Decomposition for Video 
%       Processing. https://github.com/joeblochberger/DMD/DMD_GrayTraffic.m, GitHub. Retrieved Month Day, Year.
%
% Usage:
% This script demonstrates the use of dynamic mode decomposition *DMD) to
% seperate out foreground and background components of the traffic.avi file
% avaliable in the MATLAB Image Processing Toolbox.  Note: One must
% generate the grayscale version of traffic.avi first before using this
% code.

clc; clear all; close all;
v = VideoReader('GrayTraffic.avi');
frames = read(v,[1 Inf]);
size(frames)
X=permute(AgentsOverTime,[2 3 1]);
X = squeeze(frames(:,:,1,:));
sz=size(X)

X_tall = double(reshape(X,sz(1)*sz(2),sz(3)));
t=1:sz(3);
dt = t(2) - t(1); 
n=1:sz(1)*sz(2);

%% Create data matrices for DMD
X1 = X_tall(:,1:end-1);
X2 = X_tall(:,2:end);

%% SVD and rank-50 truncation (arbitrary)
r = 50; % rank truncation

[U, S, V] = svd(X1, 'econ');
Ur = U(:, 1:r);
Sr = S(1:r, 1:r);
Vr = V(:, 1:r);

%% Build Atilde and DMD Modes
Atilde = Ur'*X2*Vr/Sr;
[W, D] = eig(Atilde);
Phi = X2*Vr/Sr*W;  % DMD Modes

%% DMD Spectra
lambda = diag(D);
omega = log(lambda)/dt;

figure;
plot(omega, '.');
title('DMD Spectra')

%% Find near-zero eigenvalue and parse out foreground eigens from background eigen
bg = find(abs(omega)<1e-2);
fg = setdiff(1:r, bg);

omega_fg = omega(fg); % foreground
Phi_fg = Phi(:,fg); % DMD foreground modes

omega_bg = omega(bg); % background
Phi_bg = Phi(:,bg); % DMD background mode

%% Compute DMD Background Solution
b = Phi_bg \ X_tall(:, 1);
X_bg = zeros(numel(omega_bg), length(t));
for tt = 1:length(t)
    X_bg(:, tt) = b .* exp(omega_bg .* t(tt));
end
X_bg = Phi_bg * X_bg;
% X_bg = X_bg(1:n, :);

figure;
X_DMD_bg=uint8(reshape(real(X_bg),sz(1),sz(2),sz(3)));
imagesc(X_DMD_bg(:,:,50))
colormap gray
daspect([1 1 1])
title('DMD Background Solution')
%% Compute DMD Foreground Solution
b = Phi_fg \ X_tall(:, 1);
X_fg = zeros(numel(omega_fg), length(t));
for tt = 1:length(t)
    X_fg(:, tt) = b .* exp(omega_fg .* t(tt));
end
X_fg = Phi_fg * X_fg;
% X_fg = X_fg(1:n, :);

figure;
X_DMD_fg=uint8(reshape(real(X_fg),sz(1),sz(2),sz(3)));
imagesc(X_DMD_fg(:,:,50))
colormap gray
daspect([1 1 1])
title('DMD Foreground Solution')
%% Compute DMD Reconstructed Solution
X_DMD=X_bg+X_fg;
figure;
X_DMD=uint8(reshape(real(X_DMD),sz(1),sz(2),sz(3)));
imagesc(X_DMD(:,:,50))
colormap gray
daspect([1 1 1])

%% Compute DMD Low-Rank
X_lowrank=X-real(X_DMD_fg);
figure;
X_lowrank=uint8(reshape(real(X_lowrank),sz(1),sz(2),sz(3)));
imagesc(X_lowrank(:,:,50))
colormap gray
daspect([1 1 1])

%% Compute DMD Sparse
X_sparse=X-real(X_DMD_bg);
figure;
X_sparse=uint8(reshape(real(X_sparse),sz(1),sz(2),sz(3)));
imagesc(X_sparse(:,:,50))
colormap gray
daspect([1 1 1])

%% Movie
v = VideoWriter('X.avi');
open(v)
for k=1:sz(3)
    imagesc(X(:,:,k))
    colormap gray
    daspect([1 1 1])
    frame=getframe(gcf);
    writeVideo(v,frame);
end
close(v)

v = VideoWriter('X_DMD_bg.avi');
open(v)
for k=1:sz(3)
    imagesc(X_DMD_bg(:,:,k))
    colormap gray
    daspect([1 1 1])
    frame=getframe(gcf);
    writeVideo(v,frame);
end
close(v)

v = VideoWriter('X-X_DMD_bg.avi');
open(v)
for k=1:sz(3)
    imagesc(X(:,:,k)-X_DMD_bg(:,:,k))
    colormap gray
    daspect([1 1 1])
    frame=getframe(gcf);
    writeVideo(v,frame);
end
close(v)

v = VideoWriter('X_lowrank.avi');
open(v)
for k=1:sz(3)
    imagesc(X_lowrank(:,:,k))
    colormap gray
    daspect([1 1 1])
    frame=getframe(gcf);
    writeVideo(v,frame);
end
close(v)

v = VideoWriter('X_sparse.avi');
open(v)
for k=1:sz(3)
    imagesc(X_sparse(:,:,k))
    colormap gray
    daspect([1 1 1])
    frame=getframe(gcf);
    writeVideo(v,frame);
end
close(v)

implay(X)
implay(X_DMD_bg)
implay(X-X_DMD_bg)
implay(X_lowrank)
implay(X_sparse)

%% frame analysis for entropy
frames = [7, 37, 49, 70, 92, 112, 120];
for ff=frames
    entropy(X(:,:,ff))
end

for ff=frames
    entropy(X_sparse(:,:,ff))
end

for ff=frames
    entropy(X_lowrank(:,:,ff))
end

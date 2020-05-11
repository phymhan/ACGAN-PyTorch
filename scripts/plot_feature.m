function plot_feature(model, ep)

result = 'results_psi';
cifar = 'cifar10';
% model = 'tac+naof';
% ep = 14;
nbatch = 4;
bs = 256;

f_fake = [];
y_fake = [];
f_real = [];
y_real = [];
for b = 0:nbatch-1
    f_fake_ = readNPY(sprintf('/media/ligong/Passport/Share/dresden/Active/ACGAN-PyTorch/%s/%s/%s/features/fake_epoch_%d_batch_%d_f.npy', result, cifar, model, ep, b));
    y_fake_ = readNPY(sprintf('/media/ligong/Passport/Share/dresden/Active/ACGAN-PyTorch/%s/%s/%s/features/fake_epoch_%d_batch_%d_y.npy', result, cifar, model, ep, b));
    f_real_ = readNPY(sprintf('/media/ligong/Passport/Share/dresden/Active/ACGAN-PyTorch/%s/%s/%s/features/real_epoch_%d_batch_%d_f.npy', result, cifar, model, ep, b));
    y_real_ = readNPY(sprintf('/media/ligong/Passport/Share/dresden/Active/ACGAN-PyTorch/%s/%s/%s/features/real_epoch_%d_batch_%d_y.npy', result, cifar, model, ep, b));
    f_fake = cat(1, f_fake, f_fake_);
    y_fake = cat(1, y_fake, y_fake_);
    f_real = cat(1, f_real, f_real_);
    y_real = cat(1, y_real, y_real_);
end

f = cat(1, f_fake, f_real);
y = cat(1, y_fake, y_real);

f2 = tsne(f);
y_disc = cat(1, zeros(bs*nbatch,1), ones(bs*nbatch,1));

figure('name', sprintf('%s epoch %d', model, ep), 'color', [1 1 1])

subplot(2, 2, 1)
gscatter(f2(:,1), f2(:,2), y_disc)
title('dis');
subplot(2, 2, 2)
gscatter(f2(:,1), f2(:,2), y)
title('cls');

f2_real = tsne(f_real);
subplot(2, 2, 3)
gscatter(f2_real(:,1), f2_real(:,2), y_real)
title('cls real');

f2_fake = tsne(f_fake);
subplot(2, 2, 4)
gscatter(f2_fake(:,1), f2_fake(:,2), y_real)
title('cls fake');

set(gcf, 'position', [800         100        1000        1000])

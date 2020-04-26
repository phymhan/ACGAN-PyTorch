cifar = 'cifar10';
model = 'tac+naof';

ep = 14;

f_fake = readNPY(sprintf('/media/ligong/Passport/Share/cbimfs/Active/ACGAN-PyTorch/results_feat/%s/%s/features/fake_epoch_%d_batch_0_f.npy', cifar, model, ep));
y_fake = readNPY(sprintf('/media/ligong/Passport/Share/cbimfs/Active/ACGAN-PyTorch/results_feat/%s/%s/features/fake_epoch_%d_batch_0_y.npy', cifar, model, ep));
f_real = readNPY(sprintf('/media/ligong/Passport/Share/cbimfs/Active/ACGAN-PyTorch/results_feat/%s/%s/features/real_epoch_%d_batch_0_f.npy', cifar, model, ep));
y_real = readNPY(sprintf('/media/ligong/Passport/Share/cbimfs/Active/ACGAN-PyTorch/results_feat/%s/%s/features/real_epoch_%d_batch_0_y.npy', cifar, model, ep));

f = cat(1, f_fake, f_real);
y = cat(1, y_fake, y_real);

f2 = tsne(f);

y_disc = cat(1, zeros(256,1), ones(256,1));

figure; gscatter(f2(:,1), f2(:,2), y_disc)
figure; gscatter(f2(:,1), f2(:,2), y)

f2_real = tsne(f_real);
figure; gscatter(f2_real(:,1), f2_real(:,2), y_real)
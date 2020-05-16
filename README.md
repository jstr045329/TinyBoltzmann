Tiny Restricted Boltzmann Machine, with contrastive divergence training. Written in Matlab. 

Run testTinyBoltzmann.m to see your RBM train on a tiny test set. 

After that, type:

    uut = uut.pingPongTest(1, img_02, 10)

to bounce a test image back & forth between layers 1 & 2 several times. Then try:

    heatmap(reshape(uut.layers{1}, 15, 15))

to see a picture of your reconstructed image. It should look like a tall skinny rectangle.
Note this last step requires heatmap.m, which you can copy & paste off the Mathworks website.
I Googled "code heatmap.m" to find it. It's a handy file to have in genernal. 


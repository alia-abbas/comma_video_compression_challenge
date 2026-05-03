
model takes two inputs:
1. time embedding: small vector of numbers unique to the specific frame index 
2. coordinate grid: a fixed 2D map of (x, y) coordinates representing every pixel position in the frame

the models is series of 1x1 convolutions. there act as MLP that processes every pixel in the grid simultaneoulsy.
it concatenates time embeddings w the spatial coordinates. this combined data passes through nn layers

output - nn preforms a regressino to predict a specific (R,G,B) color value for every coordainte at the specific time

Compression: instead of storing pixels, you store the weights of the nn and the time embeddings. because these weights are shared across al 1200 frames, the model 'memorizes' the repeating patterns of the video (like the road and the sky) very efficiently.


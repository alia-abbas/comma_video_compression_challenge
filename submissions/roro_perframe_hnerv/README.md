# A per frame neural codec for the comma compression challenge

## What the challenge actually rewards

The task hands you one 60 second dashcam clip and two frozen networks. A SegNet that segments the road scene and a PoseNet that estimates ego motion between consecutive frames. The score adds three parts. One hundred times the SegNet class disagreement, twenty five times the compressed size over the original, and the square root of ten times the PoseNet error. Lower wins. Roughly half of a strong score is the file size and the other half is keeping the two judges happy.

I treated this as a single video overfitting problem. Rather than storing pixels, I train a small decoder network to regenerate all 1200 frames, then store only that network plus a short code for each frame. At playback the decoder redraws every frame. The decoder is the bulk of the archive, so most of the work went into making it both faithful and tiny.

## The architecture, and why it is different

The leading entries use one latent code per frame pair feeding a decoder with two output heads. I went the other way. Every frame gets its own latent vector, and a single shared dense decoder turns each latent into one frame at the resolution the judges actually consume (384 by 512), which I then bicubic upsample to camera resolution. Per frame codes cost a little more in size, but they let the reconstruction carry true frame to frame motion, which turned out to matter enormously for the PoseNet term.

The decoder is a NeRV style stack of about 230 thousand parameters. A linear stem expands the latent into a small feature grid, then six upsampling blocks (full 3 by 3 convolutions, pixel shuffle, sinusoidal activations, bilinear skip connections) take it up to full resolution.

## The bug that was hiding the whole problem

For a long time my PoseNet error refused to move. The SegNet term trained fine, but pose sat frozen no matter how heavily I weighted it. The reason was subtle. PoseNet preprocesses its input through an rgb to yuv conversion that the challenge code wraps in a no gradient context and writes in place. That severs the autograd graph, so the pose loss had a gradient of exactly zero reaching my decoder. Pose was never actually being trained. It only drifted as a side effect of pixel fidelity.

The fix was a value identical but differentiable replacement for that conversion, patched in only during training. The instant the gradient could flow, pose collapsed from about 2.5 to under 0.15 in twenty epochs. This was the single most important discovery in the whole effort, and it is the thing that turned a worse than baseline result into a competitive one.

## The training recipe

I trained in two phases. First a pixel fidelity pretrain so the decoder could reproduce each frame at all. Then a metric space fine tune where the real objective takes over. A smooth disagreement loss for the SegNet, which puts the strongest gradient on the boundary pixels that are about to flip class (exactly what the argmax metric counts), and a concave square root form for the PoseNet error, which keeps a useful gradient even when the error is already small, where plain mean squared error flatlines.

Two tricks stabilized the notoriously noisy single video overfit. An exponential moving average of the weights, which I evaluate and ship instead of the raw noisy weights, and a best checkpoint tracker that keeps the lowest scoring state ever seen. I also moved the hidden convolution weights onto a Newton Schulz orthogonalized momentum optimizer, which gave a clean extra drop late in training. Weights are quantized to INT6 per tensor and then brotli compressed, with quantization aware training from the start so the shipped integer weights match what the network learned.

## Rate versus distortion, and where the floor is

Once distortion was low, the file size term dominated, so I ran the obvious size experiments. They taught me something useful. INT5 weights shrink the file but wreck pose, because the weights were never adapted to the coarser grid. Magnitude pruning shrinks the file too, but after a fine tune to recover, the distortion climbs back by about as much as the size dropped. Both are a wash, and for the same reason. The model is at capacity. Almost every weight is doing real work, so trading bits for distortion runs close to one for one. INT6 sits at the sweet spot.

## Result

The submission scores 0.48 on the public evaluator. SegNet disagreement around 0.0025, PoseNet error around 0.0014, and a 164 kilobyte archive. The progression went from worse than the trivial baseline all the way down, with the pose gradient fix as the clear turning point and the metric space losses plus the moving average doing the rest.

## What I would try next

The remaining gap to the very top is entirely distortion, not file size, since my rate is already competitive. SegNet disagreement is the stubborn term. Pushing it lower would mean a much longer staged training schedule than I ran here, closer to the scale the top entries used. The honest read is that the last stretch is bought mostly with training hours rather than one more clever trick.

## Files

`inflate.sh` and `inflate.py` decode `archive.zip` back to raw frames. `src/model.py` is the decoder, `src/codec.py` is the quantizer and entropy coder, `src/train.py` is the full training pipeline, and `compress.sh` reproduces the archive from scratch.

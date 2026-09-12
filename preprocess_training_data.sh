
# Descriptions of parameters:

# linearize: If the input images should be linearized.
# imsize: Dimensions of images to generate - width, height, channels.
# input_path: Path to raw, unprocessed data.
# output_path: Path to processed data, ready to be used for training.
# subimages: Number of augmentations to generate per image.
# cropscale: Dimensions of cropped subimages - width, height.
# clip: Min and max percentage of intensities to maintain (the rest will be clipped in the LDR).
# noise: Noise std range to pick from.
# hue: Hue mean and std.
# sat: Color saturation mean and std.
# sigmoid_n: Sigmoidal parameter 'n' mean and std.
# sigmoid_a: Sigmoidal parameter 'a' mean and std.
# jpeg_quality: Minimum JPEG quality

./virtualcamera/virtualcamera -linearize 0 -imsize 320 320 3 -input_path path/to/input -output_path path/to/output -subimages 1 -cropscale 0.2 0.6 -clip 0.98 0.99 -noise 0.0 0.01 -hue 0.0 7.0 -sat 0.0 0.1 -sigmoid_n 0.9 0.1 -sigmoid_a 0.6 0.1 -jpeg_quality 30 --verbose

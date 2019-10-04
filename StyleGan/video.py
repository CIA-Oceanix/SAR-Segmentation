import pickle
import os

import PIL.Image
import dnnlib.tflib as tflib
import numpy as np
import fire

from Rignak_Misc.path import get_local_file

MODEL_FILENAME = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
RESULT_ROOT = get_local_file(__file__, 'results')
TRUNCATION_PSI = 0.7
GRID_DIM = 2
DURATION = 60
FPS = 20
IMAGE_ZOOM = 1


def main(model_filename=MODEL_FILENAME, truncation_psi=TRUNCATION_PSI, result_root=RESULT_ROOT,
         duration=DURATION, grid_dim=GRID_DIM, fps=FPS, image_zoom=IMAGE_ZOOM):
    """
    Create an interpolation video

    :param model_filename: name of the file containing the models
    :param truncation_psi: originality factor, closer to 0 means less original
    :param result_root: name of the folder to contains the output
    :param duration: length, in seconds, of the video to create
    :param grid_dim: number of thumbnails on each row and columns of the video
    :param fps: frame per seconds
    :param image_zoom: zoom applied on the output
    :return:
    """
    tflib.init_tf()
    _G, _D, Gs = pickle.load(open(model_filename, "rb"))

    grid_size = [grid_dim, grid_dim]
    smoothing_sec = 1.0
    mp4_codec = 'libx264'
    mp4_bitrate = '5M'
    mp4_file = os.path.join(result_root, os.path.splitext(os.path.split(model_filename)[-1])[0] + '.mp4')

    num_frames = int(np.rint(duration * fps))

    # Generate latent vectors
    shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:]  # [frame, image, channel, component]
    all_latents = np.random.randn(*shape).astype(np.float32)
    import scipy
    all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * fps] + [0] * len(Gs.input_shape),
                                                mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    def create_image_grid(images, grid_size=None):
        assert images.ndim == 3 or images.ndim == 4
        num, img_h, img_w, channels = images.shape

        if grid_size is not None:
            grid_w, grid_h = tuple(grid_size)
        else:
            grid_w = max(int(np.ceil(np.sqrt(num))), 1)
            grid_h = max((num - 1) // grid_w + 1, 1)

        grid = np.zeros([grid_h * img_h, grid_w * img_w, channels], dtype=images.dtype)
        for idx in range(num):
            x = (idx % grid_w) * img_w
            y = (idx // grid_w) * img_h
            grid[y: y + img_h, x: x + img_w] = images[idx]
        return grid

    # Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7,
                        randomize_noise=False, output_transform=fmt)

        grid = create_image_grid(images, grid_size)
        if image_zoom > 1:
            grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2)  # grayscale => RGB
        return grid

    # Generate video.
    import moviepy.editor
    video_clip = moviepy.editor.VideoClip(make_frame, duration=duration)
    video_clip.write_videofile(mp4_file, fps=fps, codec=mp4_codec, bitrate=mp4_bitrate)

    duration = 60.0
    smoothing_sec = 1.0
    fps = 20

    num_frames = int(np.rint(duration * fps))
    random_seed = 500
    random_state = np.random.RandomState(random_seed)

    w = 512
    h = 512
    dst_seeds = [700]
    style_ranges = ([0] * 7 + [range(8, 16)]) * len(dst_seeds)

    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    synthesis_kwargs = dict(output_transform=fmt, truncation_psi=truncation_psi, minibatch_size=8)

    shape = [num_frames] + Gs.input_shape[1:]
    src_latents = random_state.randn(*shape).astype(np.float32)
    src_latents = scipy.ndimage.gaussian_filter(src_latents,
                                                smoothing_sec * fps,
                                                mode='wrap')
    src_latents /= np.sqrt(np.mean(np.square(src_latents)))

    dst_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds)

    src_dlatents = Gs.components.mapping.run(src_latents, None)
    dst_dlatents = Gs.components.mapping.run(dst_latents, None)
    src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
    dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **synthesis_kwargs)

    canvas = PIL.Image.new('RGB', (w * (len(dst_seeds) + 1), h * 2), 'white')

    for col, dst_image in enumerate(list(dst_images)):
        canvas.paste(PIL.Image.fromarray(dst_image, 'RGB'), ((col + 1) * h, 0))

    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * fps), 0, num_frames - 1))
        src_image = src_images[frame_idx]
        canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), (0, h))

        for col, dst_image in enumerate(list(dst_images)):
            col_dlatents = np.stack([dst_dlatents[col]])
            col_dlatents[:, style_ranges[col]] = src_dlatents[frame_idx, style_ranges[col]]
            col_images = Gs.components.synthesis.run(col_dlatents, randomize_noise=False, **synthesis_kwargs)
            for row, image in enumerate(list(col_images)):
                canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * h, (row + 1) * w))
        return np.array(canvas)

    # Generate video.
    import moviepy.editor
    mp4_file = 'results/interpolate.mp4'
    mp4_codec = 'libx264'
    mp4_bitrate = '5M'

    video_clip = moviepy.editor.VideoClip(make_frame, duration=duration)
    video_clip.write_videofile(mp4_file, fps=fps, codec=mp4_codec, bitrate=mp4_bitrate)

    import scipy

    duration = 60.0
    smoothing_sec = 1.0
    fps = 20

    num_frames = int(np.rint(duration * fps))
    random_seed = 503
    random_state = np.random.RandomState(random_seed)

    w = 512
    h = 512
    style_ranges = [range(6, 16)]

    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    synthesis_kwargs = dict(output_transform=fmt, truncation_psi=0.7, minibatch_size=8)

    shape = [num_frames] + Gs.input_shape[1:]  # [frame, image, channel, component]
    src_latents = random_state.randn(*shape).astype(np.float32)
    src_latents = scipy.ndimage.gaussian_filter(src_latents,
                                                smoothing_sec * fps,
                                                mode='wrap')
    src_latents /= np.sqrt(np.mean(np.square(src_latents)))

    dst_latents = np.stack([random_state.randn(Gs.input_shape[1])])

    src_dlatents = Gs.components.mapping.run(src_latents, None)  # [seed, layer, component]
    dst_dlatents = Gs.components.mapping.run(dst_latents, None)  # [seed, layer, component]

    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * fps), 0, num_frames - 1))
        col_dlatents = np.stack([dst_dlatents[0]])
        col_dlatents[:, style_ranges[0]] = src_dlatents[frame_idx, style_ranges[0]]
        col_images = Gs.components.synthesis.run(col_dlatents, randomize_noise=False, **synthesis_kwargs)
        return col_images[0]

    # Generate video.
    import moviepy.editor
    mp4_file = 'results/fine_%s.mp4' % (random_seed)
    mp4_codec = 'libx264'
    mp4_bitrate = '5M'

    video_clip = moviepy.editor.VideoClip(make_frame, duration=duration)
    video_clip.write_videofile(mp4_file, fps=fps, codec=mp4_codec, bitrate=mp4_bitrate)


if __name__ == "__main__":
    fire.Fire(main)

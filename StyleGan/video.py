import pickle
import os
import scipy
import moviepy.editor

import PIL.Image
import dnnlib.tflib as tflib
import numpy as np
import fire

from Rignak_Misc.path import get_local_file

MODEL_FILENAME = 'sar_network-snapshot-009075.pkl'
MODE = ['noisy', 'noiseless', 'fixed_noise', 'fixed_latent'][-1]
RESULT_ROOT = get_local_file(__file__, 'results')
TRUNCATION_PSI = 0.7
GRID_DIM = 2
DURATION = 60
FPS = 20
IMAGE_ZOOM = 1


def main(model_filename=MODEL_FILENAME, truncation_psi=TRUNCATION_PSI, result_root=RESULT_ROOT,
         duration=DURATION, grid_dim=GRID_DIM, fps=FPS, image_zoom=IMAGE_ZOOM, mode=MODE):
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
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

        randomize_noise = False
        use_noise = True
        latents = all_latents[frame_idx]
        if mode == 'noisy':
            randomize_noise = True
        elif mode == 'noiseless':
            use_noise = False
        elif mode == 'fixed_latent':
            latents = all_latents[0]
            randomize_noise = True

        images = Gs.run(latents, None, truncation_psi=truncation_psi,
                        randomize_noise=randomize_noise, use_noise=use_noise, output_transform=fmt)

        grid = create_image_grid(images, grid_size)
        if image_zoom > 1:
            grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2)  # grayscale => RGB
        return grid

    tflib.init_tf()
    _G, _D, Gs = pickle.load(open(model_filename, "rb"))

    grid_size = [grid_dim, grid_dim]
    smoothing_sec = 1.0
    mp4_codec = 'libx264'
    mp4_bitrate = '5M'
    mp4_file = os.path.join(result_root, f'{os.path.splitext(os.path.split(model_filename)[-1])[0]}_{mode}.mp4')

    num_frames = int(np.rint(duration * fps))

    # Generate latent vectors
    shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:]  # [frame, image, channel, component]
    all_latents = np.random.randn(*shape).astype(np.float32)
    all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * fps] + [0] * len(Gs.input_shape),
                                                mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    # Generate video.
    video_clip = moviepy.editor.VideoClip(make_frame, duration=duration)
    video_clip.write_videofile(mp4_file, fps=fps, codec=mp4_codec, bitrate=mp4_bitrate)


if __name__ == "__main__":
    fire.Fire(main)

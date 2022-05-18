import numpy as np

def gan_im2im_generator(generator):
	while True:
		input_batch, output_batch = next(generator)
		new_input_batch = np.zeros(list(input_batch.shape[:3]) + [input_batch.shape[3]+1])
		new_input_batch[:,:,:,:input_batch.shape[3]] = input_batch
		new_input_batch[:,:,:,-1] = np.random.random(new_input_batch.shape[:-1])
		yield new_input_batch, output_batch

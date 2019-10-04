# choose the dataset
dataset                 = 'faces'

# choose to resume or not
resume_run_id           = 2 # none : from scratch
resume_kimg             = 140

# doubling the resolution every x kimg
lod_training_kimg       = 600

# when to do the ticks
tick_kimg_base          = 160
tick_kimg_dict          = {4: 160, 8:140, 16:120, 32:100, 64:80, 128:60, 256:40}

#when to do the snapshot
image_snapshot_ticks    = 1
network_snapshot_ticks  = 10
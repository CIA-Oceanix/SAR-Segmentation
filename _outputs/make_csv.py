import os
import json
import glob
import tqdm
import pathlib
from datetime import datetime

filenames = glob.glob(os.path.join('.', '**' ,'*.json'), recursive=True)
filenames = [filename for filename in filenames if not filename.endswith('_samples.json')]

models = {}

keys = []
for filename in tqdm.tqdm(filenames):
	models[filename] = {}
	models[filename]['_id'] = filename
	
	with open(filename, 'r') as json_file:
		model_json = json.load(json_file)
		models[filename]['task'] = model_json.get('_config', {}).get('TASK')
		models[filename]['input_shape'] = model_json.get('_config', {}).get('INPUT_SHAPE')
		models[filename]['dataset'] = model_json.get('_config', {}).get('DATASET')
		models[filename]['val_loss'] = min([v.get('val_loss') for v in model_json.get("_logs", [{}])])
		models[filename]['val_accuracy'] = max([v.get('val_accuracy', 0) for v in model_json.get("_logs", [{}])])
	models[filename]['last_modified'] = datetime.fromtimestamp(pathlib.Path(filename).stat().st_mtime).strftime('%Y-%m-%d')
		
	for key in models[filename].keys():
		if key not in keys:
			keys.append(key)
			
keys = sorted(list(set(keys)))

lines = ['\t'.join(keys)]
for model in models.values():
	lines.append('\t'.join([repr(model.get(key)) for key in keys]))

with open('aggregation.txt', 'w') as file:
	for line in lines:
		while '\\\\' in line:
			line = line.replace('\\\\', '/')
		while '\''  in line:
			line = line.replace('\'', '')
		while '//'  in line:
			line = line.replace('//', '/')
		file.write(line + '\n')

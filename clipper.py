from PIL import Image

def ResNet50(inp):
	import numpy as np
	# from PIL import Image
	import requests
	import torch
	import pandas as pd
	import torchvision
	labels = {
		int(key): value
		for (key, value) in requests.get(
			"https://s3.amazonaws.com/outcome-blog/imagenet/labels.json"
		)
		.json()
		.items()
	}
	preprocess = torchvision.transforms.Compose(
    	[
        	torchvision.transforms.Resize(256),
        	torchvision.transforms.CenterCrop(224),
        	torchvision.transforms.ToTensor(),
        	torchvision.transforms.Normalize(
            	mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        	),
    	]
	)

	model50 = torchvision.models.resnet50(pretrained=True)
	model50.eval()

	input_tensor = preprocess(inp)
	input_batch = input_tensor.unsqueeze(0)
	with torch.no_grad():
		output = model50(input_batch)
	proba = torch.nn.functional.softmax(output[0], dim=0).numpy()
	top3 = np.argsort(proba)[-3:][::-1]
	l = [labels[i] for i in top3]
	probs = [proba[i] for i in top3]
	df = pd.DataFrame({"rank": [1, 2, 3], "probability": probs, "category": l}).astype(
		str
	)
	return df

img = Image.open("dog_test.jpg")
print(ResNet50(img))

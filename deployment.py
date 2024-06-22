import runpod
from run_inference import image_enhance

runpod.serverless.start({"handler": image_enhance})

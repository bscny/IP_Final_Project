import kagglehub, os
os.environ["KAGGLEHUB_CACHE_DIR"] = "./data/div2k"
path = kagglehub.dataset_download("soumikrakshit/div2k-high-resolution-images")
print("Path to dataset files:", path)

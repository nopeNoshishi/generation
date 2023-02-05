from PIL import Image
import cupy as cp


images = []

# main image
img = Image.open("kabin.jpg")

# 256 * 256 * 3
start_image = img.resize((256, 256))

# Normalize [0 ~ 1]
x = cp.array(start_image) / 255

# hyper parameter
time = 1000
beta_start = 10e-4
beta_end = 10e-3 * 2
betas = []
for i in range(time):
    betas.append(beta_start + ((beta_end - beta_start) / 1000) * i) # Linear scheling

alpha_ = 1
for beta in betas:
    alpha = 1 - beta
    alpha_ *= alpha

# Noising
x_start = x

noise = cp.random.normal(
    loc=0,
    scale=1,
    size=x_start.shape
)

x_end = cp.sqrt(alpha_) * x_start + cp.sqrt(1.0 - alpha_) * noise

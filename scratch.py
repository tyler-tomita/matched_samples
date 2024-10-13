import torch
import torchvision
from torchvision.transforms import ToTensor, ToPILImage
from diffusers import StableDiffusionPipeline, AutoPipelineForImage2Image, StableDiffusionUpscalePipeline
from diffusers.utils import load_image, make_image_grid
from matplotlib import pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from torch import fft

device = torch.device('mps')

dataset_dir = '/Users/tyler/datasets/CelebA'

celeba_data = torchvision.datasets.CelebA(
    dataset_dir,
    split='train',
    target_type='attr',
    transform=None,
    download=False
)

data_loader = torch.utils.data.DataLoader(celeba_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=8)


# pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')
pipe = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
# pipe = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16)
pipe.safety_checker = None
pipe.requires_safety_checker = False
pipe = pipe.to(device)

vae = pipe.vae
unet = pipe.unet
noise_scheduler = pipe.scheduler

# init_image = celeba_data[0][0]

init_image = Image.open('cat/1.jpeg')

if not init_image.mode == "RGB":
    init_image = init_image.convert("RGB")

init_image = init_image.resize((512, 512), resample=Image.Resampling.BICUBIC)

init_image = ToTensor()(init_image).unsqueeze(0)
init_image = init_image.to(device=device, dtype=torch.float16)

# autoencode image
vae.eval()
with torch.no_grad():
    recon_image = vae(init_image)[0]

recon_image_pil = ToPILImage()(recon_image[0])
init_image_pil = ToPILImage()(init_image[0])
image_grid = make_image_grid([init_image_pil.resize(recon_image_pil.size), recon_image_pil], rows=1, cols=2)
plt.imshow(image_grid)
plt.show()

### encode images in latent space ###
with torch.no_grad():
    latents = vae.encode(init_image).latent_dist.sample().detach()
# latents = latents * vae.config.scaling_factor

### decode to original image space ###
with torch.no_grad():
    recon_image = vae.decode(latents).sample

# prompt = "a white cat"
prompt = "a woman"
image = pipe(prompt=prompt, image=init_image, strength=0.005, num_inference_steps=20000).images[0]
# image = pipe(prompt=prompt, image=low_res_img).images[0]

# inspect image
init_image = ToPILImage()(init_image[0])
# image = ToPILImage()(image[0])
# image_grid = make_image_grid([init_image.resize(image.size), image], rows=1, cols=2)

recon_image_cpu = recon_image.to(dtype=torch.float32, device="cpu")
recon_image.requires_grad = True
init_image_cpu = init_image.to(dtype=torch.float32, device="cpu")
init_image.requires_grad = True
recon_image_fft = fft.fft(recon_image_cpu)
init_image_fft = fft.fft(init_image_cpu)
image_grid = make_image_grid([init_image_pil, recon_image_pil, init_image_fft, recon_image_fft], rows=2, cols=2)
# image_grid = make_image_grid([init_image.resize(recon_image.size), recon_image, init_image_fft.resize(recon_image_fft.size), recon_image_fft], rows=2, cols=2)

plt.imshow(image_grid)
plt.show()


url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
response = requests.get(url)
init_img = Image.open(BytesIO(response.content)).convert("RGB")
low_res_img = init_img.resize((32, 32), resample=Image.LANCZOS)
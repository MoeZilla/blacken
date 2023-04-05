from kandinsky2 import get_kandinsky2
model = get_kandinsky2('cuda', task_type='text2img', model_version='2.1', use_flash_attention=False)

images = model.generate_text2img(
    "red cat, 4k photo", 
    num_steps=100,
    batch_size=1, 
    guidance_scale=4,
    h=768, w=768,
    sampler='p_sampler', 
    prior_cf_scale=4,
    prior_steps="5"
)

print(images)
from fastapi import FastAPI
from typing import List, Optional, Union
import io, uvicorn, gc
from fastapi.responses import StreamingResponse
import torch
import time
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
app.POOL: ThreadPoolExecutor = None

@app.on_event("startup")
def startup_event():
    app.POOL = ThreadPoolExecutor(max_workers=1)
@app.on_event("shutdown")
def shutdown_event():
    app.POOL.shutdown(wait=False)

model_id = "stabilityai/stable-diffusion-2-1"
pipe_nsd = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe_nsd.scheduler = DPMSolverMultistepScheduler.from_config(pipe_nsd.scheduler.config)
pipe_nsd = pipe_nsd.to("cuda")

@app.post("/getimage_nsd")
def get_image_nsd(
    #prompt: Union[str, List[str]],
    prompt: Optional[str] = "dog",
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: Optional[int] = 50,
    guidance_scale: Optional[float] = 7.5,
    negative_prompt: Optional[str] = None,):

    image = app.POOL.submit(pipe_nsd,prompt,height,width,num_inference_steps,guidance_scale,negative_prompt).result().images
    gc.collect()
    torch.cuda.empty_cache()
    filtered_image = io.BytesIO()
    image[0].save(filtered_image, "JPEG")
    filtered_image.seek(0)
    return StreamingResponse(filtered_image, media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)

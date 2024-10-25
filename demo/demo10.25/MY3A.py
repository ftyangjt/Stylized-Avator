#IMG2IMG图生图程序


import torch
import requests
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
import os
from PIL import Image

# We'll be exploring a number of pipelines today!
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline, 
    StableDiffusionDepth2ImgPipeline,
    DPMSolverMultistepScheduler,
    )       

def test(inuput_picture_address, prompt_in=None, negative_prompt_in=None, guidance_scale_in=8, steps_in=35, strength_in=0.6, seed_in=1):
    
    prompt = prompt_in
    negative_prompt = negative_prompt_in
    guidance_scale = guidance_scale_in
    init_image = Image.open(inuput_picture_address)
    steps = steps_in
    strength = strength_in
    seed = seed_in
    model_id = "./models/models--A--sd-v1-4"
    vae = 0

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(device)




    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, 
        revision="fp16",
        torch_dtype=torch.float16,
        cache_dir="./models/",
        use_safetensors=True,
        ).to(device)
        
    #pipe.enable_xformers_memory_efficient_attention() #用于增强性能
    #这里构建了流水线

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    #这里设置调度器，这里应该可以被选择
    generator = torch.Generator(device=device).manual_seed(seed) 
    #随机种子和输入种子功能应该有

    #这是一个压缩图片程序，防止我的电脑显存不足报错，用户应该可以选择想要的压缩程度
    def compress_if_large(init_image, max_size=(720, 720)):

        original_width, original_height = init_image.size
        # 检查图片是否超过指定尺寸
        if original_width > max_size[0] or original_height > max_size[1]:   
            init_image.thumbnail(max_size)                                  #thumbnail是等比压缩方法，包含于从PIL库中import的Image
            print(f"图片已压缩为: {init_image.size}")
        else:
            print("图片尺寸合适，无需压缩。")

    compress_if_large(init_image)


    # Run the pipeline, 输出
    pipe_output = pipe(
        prompt=prompt,  #这个应该有输入框
        negative_prompt=negative_prompt,  # 这个也是
        #height=512, width=512,     # 图片大小，在此处不会生效
        guidance_scale=guidance_scale,          # 决定被prompt限制程度，应该有滑块
        num_inference_steps=steps,    # 采样步数，在五十以内足够，需要有滑块
        generator=generator,       # Fixed random seed，决定种子
        strength=strength,             #更改强度，需要有滑块调整
        image=init_image
    )
    print(f"----测试输出：")
    print(f"prompt：{prompt}\nnegative_prompt：{negative_prompt}\n指导程度：{guidance_scale}\n采样步数：{steps}\n更改强度：{strength}\n种子：{seed}")
    #pipe_output.images[0].save("./result_test/A.png")
    #暂时保存
    return(pipe_output.images[0])
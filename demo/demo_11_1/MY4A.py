#IMG2IMG图生图程序

import bz2
import torch
import requests
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
import os
from PIL import Image
from compel import Compel

from preprocessor import (
    get_depth_map,
    get_array_map,
    create_face_mask,
    blur_mask    
    )

from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline, 
    StableDiffusionDepth2ImgPipeline,
    DPMSolverMultistepScheduler,
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    AutoPipelineForInpainting,
    )       
          

def test(inuput_picture_address,
        prompt_in=None,
        negative_prompt_in=None, 
        guidance_scale_in=8, 
        steps_in=35, 
        strength_in=0.6,
        seed_in=1, 
        control_net_kind=None,
        lora = None,
        face_mask = None
        ):
    
    
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(device)

    prompt = prompt_in
    negative_prompt = negative_prompt_in
    guidance_scale = guidance_scale_in
    init_image = Image.open(inuput_picture_address)
    steps = steps_in
    strength = strength_in
    seed = seed_in
    #model_id = "./models/models--stable-diffusion-v1-5--stable-diffusion-v1-5"
    model_id = "./models/artStyleXXX_v10"
    generator = torch.Generator(device=device).manual_seed(seed)



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





    def add_scheduler():
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    def add_lora():
        lora = 1
        #Lora
        #lora = "./Lora_models/fechin.safetensors"
        #monet_v2-000004.safetensors
        if lora == 1:
            lora = "./Lora_models/slg_v30.safetensors"
            lora_scale = 1.0
            pipe.load_lora_weights(lora,adapter_name="Lora1")
            adapter_weight_scales = {
                "unet":{
                    "down":0.5,
                    "mid":0.5,
                    "up":0.5
                }             
                                    }

            #pipe.set_adapters("Lora1", adapter_weight_scales)
            pipe.set_adapters("Lora1")

    #promt预处理，使用户可以在词后加入+或-改变权重
    def compel_prompt(prompt):   
        compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
        prompt_embeds = compel_proc(prompt)
        return prompt_embeds    






    #检测控制网选择类型，构建管道, 生成图片
    if control_net_kind==None and face_mask==None:

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, 
            revision="fp16",
            torch_dtype=torch.float16,
            cache_dir="./models/",
            use_safetensors=True,
            safety_checker = None
            ).to(device)
        
        add_scheduler()
        add_lora()
        prompt_embeds = compel_prompt(prompt)

        pipe_output = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt=negative_prompt,  # 这个也是
        #height=512, width=512,     # 图片大小，在此处不会生效
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator,
        strength=strength,
        image=init_image
        #cross_attention_kwargs={"scale":lora_scale}
    )
        print("1")

    elif control_net_kind==None and face_mask!=None:



        pipe = AutoPipelineForInpainting.from_pretrained(
        model_id, 
        revision="fp16",
        torch_dtype=torch.float16,
        cache_dir="./cache/",
        use_safetensors=True,
        safety_checker = None
        ).to(device)

        mask = create_face_mask(init_image, face_mask)
        blurred_mask = blur_mask(mask)
        
        add_scheduler()
        add_lora()
        prompt_embeds = compel_prompt(prompt)

        pipe_output = pipe(
        prompt_embeds=prompt_embeds,  
        negative_prompt=negative_prompt,  
        #height=512, width=512,     # 图片大小，在此处不会生效
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator,
        strength=strength,
        image=init_image,
        mask_image = blurred_mask,
        )
        print("2")
        
    elif control_net_kind=="dpt" and face_mask==None:
            
            depth_image = get_depth_map(init_image)

            net_model_address = "./ControlNet_models/models--lllyasviel--control_v11f1p_sd15_depth"
            controlnet = ControlNetModel.from_pretrained(net_model_address, torch_dtype=torch.float16, use_safetensors=True, cache_dir="./ControlNet_models").to(device)

            pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id, 
            revision="fp16",
            torch_dtype=torch.float16,
            cache_dir="./models/",
            use_safetensors=True,
            controlnet=controlnet,
            safety_checker = None
            ).to(device)

            add_scheduler()
            add_lora()
            prompt_embeds = compel_prompt(prompt)

            pipe_output = pipe(
            prompt_embeds=prompt_embeds,  
            negative_prompt=negative_prompt,  
            #height=512, width=512,     # 图片大小，在此处不会生效
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
            strength=strength,
            image=depth_image,
            controlnet_conditioning_scale=1.0,
            #control_guidance_start=0.0,
            #control_guidance_end=1.0,
            )
            print("3")

    elif control_net_kind=="array" and face_mask==None:
        array_image = get_array_map(init_image)

        net_model_address = "./ControlNet_models/models--lllyasviel--control_v11f1p_sd15_depth"
        controlnet = ControlNetModel.from_pretrained(net_model_address, torch_dtype=torch.float16, use_safetensors=True, cache_dir="./ControlNet_models").to(device)

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id, 
        revision="fp16",
        torch_dtype=torch.float16,
        cache_dir="./models/",
        use_safetensors=True,
        controlnet=controlnet,
        safety_checker = None
        ).to(device)


        add_scheduler()
        add_lora()
        prompt_embeds = compel_prompt(prompt)

        pipe_output = pipe(
        prompt_embeds=prompt_embeds,  
        negative_prompt=negative_prompt,  
        #height=512, width=512,     # 图片大小，在此处不会生效
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator,
        strength=strength,
        image=array_image,
        controlnet_conditioning_scale=1.0,
        #control_guidance_start=0.0,
        #control_guidance_end=1.0,
            )
        print("4")
    
        

    








    print(f"----本次测试输出：")
    print(f"模型文件:{model_id}\nprompt：{prompt}\nnegative_prompt：{negative_prompt}\n指导程度：{guidance_scale}\n采样步数：{steps}\n更改强度：{strength}\n种子：{seed}")
    #pipe_output.images[0].save("./result_test/A.png")
    #暂时保存
    return(pipe_output.images[0])
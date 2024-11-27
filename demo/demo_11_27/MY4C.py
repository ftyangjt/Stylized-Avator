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
    blur_mask,
    face_repairer, 
    portrait_mask,
    background_worker,   
    add_random_spots,
    distort_image,
    background_replace,
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
    AutoencoderKL,
    StableDiffusionControlNetInpaintPipeline,
    )       
          

def build_pipeline(
        inuput_picture_address,
        prompt_in,
        negative_prompt_in, 
        guidance_scale_in, 
        steps_in, 
        strength_in,
        seed_in, 
        control_net_kind,
        face_mask,
        in_lora_dict,
        model,
        ):
    
    
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(device)

    model_dict={
        "ArtStyle":"./models/artStyleXXX_v10",
        "stable-diffusion-v1-5":"./models/stable-diffusion-v1-5",
        "majicMIX realistic":"./models/majicMIX realistic",
        "majicMIX realistic":"./models/majicMIX realistic"
    }

    model_id = model_dict[model]
    


    def add_scheduler():
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    #未集成功能
    def add_lora_0():
        lora = 1
        
        #Lora
        #lora = "./Lora_models/fechin.safetensors"
        #monet_v2-000004.safetensors
        if lora == 1:
            #lora = "./Lora_models/anime_minimalist_v1-000020"
            lora = "./Lora_models/slg_v30.safetensors"
            lora_2 = "./Lora_models/Artista V2.safetensors"
            lora_2 = "./Lora_models/Hand v3 SD1.5.safetensors"
            lora_scale = 1.0
            pipe.load_lora_weights(lora,adapter_name="Lora1")
            #pipe.load_lora_weights(lora_2,adapter_name="Lora2")
            pipe
            adapter_weight_scales = {
                "unet":{
                    "down":0.5,
                    "mid":0.5,
                    "up":0.5
                }             
                                    }

            #pipe.set_adapters("Lora1", adapter_weight_scales)
            pipe.set_adapters("Lora1")
            #pipe.set_adapters(["Lora1","Lora2"],adapter_weights=[1.0,1.0])

    def add_lora(in_lora_dict):
        if in_lora_dict:
            all_lora_dict = {
                "./Lora_models/fechin.safetensors": "fechin",
                "./Lora_models/monet_v2-000004.safetensors": "monet_v2" ,  
                "./Lora_models/slg_v30.safetensors": "森林之光",
                "./Lora_models/Pixel Art.safetensors": "Pixel Art",
                "./Lora_models/bichu-v0612.safetensors": "油画笔触" ,  
                "./Lora_models/Colorwater_v4.safetensors": "沁彩" ,  
                "./Lora_models/MoXinV1.safetensors": "墨心" ,  
                "./Lora_models/howlbgsv3.safetensors": "哈尔的移动城堡" ,  
                "./Lora_models/soulcard.safetensors": "灵魂卡" ,  
                "./Lora_models/ankymoore-04.safetensors": "ankymoore的现代艺术" ,  
                "./Lora_models/retrowave_0.12.safetensors": "retrowave  合成波艺术" ,  
                "./Lora_models/pixarStyleModel_lora128.safetensors": "皮克斯动画" ,  
                "./Lora_models/xk3mt1ks.safetensors": "Schematics  黑白底概念图" ,  
                "./Lora_models/br_max-000014.safetensors": "Silhouette Synthesis 逆光幻境" ,  
                "./Lora_models/npzw-05.safetensors": "旧报纸风格" ,  
                "./Lora_models/soviet-poster.safetensors": "苏联海报" ,  
                "./Lora_models/Night scene_20230715120543.safetensors": "华灯初上/Night scene/fantasy city Lora" ,  
            }
            
            #用于存储所有 LoRA 的名称和权重
            loaded_adapters = []
            adapter_weights = []
            
            n = 0
            for lora, value in in_lora_dict.items():
                for address, key in all_lora_dict.items():
                    if lora == key:
                        n += 1
                        adapter_name = f"Lora{n}"
                        pipe.load_lora_weights(address, adapter_name=adapter_name)
                        loaded_adapters.append(adapter_name)
                        adapter_weights.append(value)
                        print(f"{key} 载入, {value}")
            
            # 一次性设置所有加载的 LoRA 和对应的权重
            pipe.set_adapters(loaded_adapters, adapter_weights=adapter_weights)

    def add_vae(vae="./models/stable-diffusion-v1-5/vae"):
        pipe.vae = AutoencoderKL.from_pretrained(vae, subfolder="subfolder", torch_dtype=torch.float16, revision="fp16").to("cuda")






    #检测控制网选择类型，构建管道, 生成图片
    if control_net_kind==None and face_mask["mask_class"]==None:

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, 
            revision="fp16",
            torch_dtype=torch.float16,
            cache_dir="./models/",
            use_safetensors=True,
            safety_checker = None
            ).to(device)
        
        add_scheduler()
        add_lora(in_lora_dict)
        #add_vae("./models/artStyleXXX_v10/vae")
        #IP%%%
        #pipe.load_ip_adapter("./Ip_adapter_models/ip-adapter_sd15_light", subfolder="models", weight_name="ip-adapter_sd15_light.safetensors",cache_dir="./Ip_adapter_models") 
        print("1:原始图生图管道")

    elif control_net_kind==None and face_mask["mask_class"]!=None:

        pipe = AutoPipelineForInpainting.from_pretrained(
        model_id, 
        revision="fp16",
        torch_dtype=torch.float16,
        cache_dir="./cache/",
        use_safetensors=True,
        safety_checker = None
        ).to(device)

        add_scheduler()
        add_lora(in_lora_dict)
        #add_vae("./models/artStyleXXX_v10/vae")
        print("2:人脸蒙版管道")
        
    elif control_net_kind=="dpt" and face_mask["mask_class"]==None:
            

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
            add_lora(in_lora_dict)
            #add_vae("./models/artStyleXXX_v10/vae")
            print("3:深度图管道")

    elif control_net_kind=="array" and face_mask["mask_class"]==None:

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
        add_lora(in_lora_dict)
        #add_vae("./models/artStyleXXX_v10/vae")
        print("4:边缘图管道")

    elif control_net_kind=="dpt" and face_mask["mask_class"]!=None:
        net_model_address = "./ControlNet_models/models--lllyasviel--control_v11f1p_sd15_depth"
        controlnet = ControlNetModel.from_pretrained(net_model_address, torch_dtype=torch.float16, use_safetensors=True, cache_dir="./ControlNet_models").to(device)

        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        model_id, 
        revision="fp16",
        torch_dtype=torch.float16,
        cache_dir="./models/",
        use_safetensors=True,
        controlnet=controlnet,
        safety_checker = None
        ).to(device)

        add_scheduler()
        add_lora(in_lora_dict)
        #add_vae("./models/artStyleXXX_v10/vae")
        #pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15_light.safetensors",cache_dir="./Ip_adapter_models")
        print("人脸蒙版加深度图管道")

        
    print(f"构建管道,模型{model}")   
    return(pipe)   
       

def generate_image(
        pipe,
        inuput_picture_address,
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

    pipe = pipe
    prompt = prompt_in
    negative_prompt = negative_prompt_in
    guidance_scale = guidance_scale_in
    init_image = Image.open(inuput_picture_address)
    steps = steps_in
    strength = strength_in
    seed = seed_in
    generator = torch.Generator(device=device).manual_seed(seed)



    #这是一个压缩图片程序，防止我的电脑显存不足报错，用户应该可以选择想要的压缩程度
    def compress_if_large(init_image, max_size=512):

        original_width, original_height = init_image.size
        # 检查图片是否超过指定尺寸
        if original_width > max_size or original_height > max_size:
            ratio = min(max_size / original_width, max_size / original_height)
            new_width = ratio * original_width
            new_height = ratio * original_height
            new_width -= new_width % 8
            new_height -= new_height % 8
            print(new_width,new_height)
            init_image.thumbnail((new_width, new_height))                                  #thumbnail是等比压缩方法，包含于从PIL库中import的Image
            print(f"图片已压缩为: {init_image.size}")
        else:
            print("图片尺寸合适，无需压缩。")

    compress_if_large(init_image)




    #promt预处理，使用户可以在词后加入+或-改变权重
    def compel_prompt(prompt):   
        compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
        prompt_embeds = compel_proc(prompt)
        return prompt_embeds    

    #人脸识别选择器
    def mask_generater(face_mask,init_image):
        if face_mask["mask_class"] == "粗略蒙版":
            mask_image = create_face_mask(init_image, 0) 
        elif face_mask["mask_class"] == "精细蒙版":
            mask_image = portrait_mask(init_image, face_mask["datailed_mask_set"])
        return mask_image


    #检测控制网选择类型，生成图片
    if control_net_kind==None and face_mask["mask_class"]==None:

        prompt_embeds = compel_prompt(prompt)
        #ip_adapter_image = Image.open("./picture/picture.jpg")
        pipe_output = pipe(
        #ip_adapter_image=ip_adapter_image,
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
        final_image = pipe_output.images[0]
        print("1")
        
    elif control_net_kind=="dpt" and face_mask["mask_class"]==None:
            
            depth_image = get_depth_map(init_image)
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
            final_image = pipe_output.images[0]
            print("3")

    elif control_net_kind=="array" and face_mask["mask_class"]==None:
        
        array_image = get_array_map(init_image)
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
        final_image = pipe_output.images[0]
        print("4")

    elif control_net_kind==None and face_mask["mask_class"]!=None:


        #bigger_mask是用于渐显效果的
        mask , _ , all_mask = mask_generater(face_mask,init_image)
        blurred_mask = mask
        #blurred_mask = blur_mask(mask)
        #blurred_mask = Image.open("./AAA/0Z.png")
        prompt_embeds = compel_prompt(prompt)

        if True:
            init_image = background_replace(init_image, all_mask)
            
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
        final_image = pipe_output.images[0]
        print("2")

    elif control_net_kind=="dpt" and face_mask["mask_class"]!=None:

        #mask = create_face_mask(init_image, face_mask)        
        #blurred_mask = blur_mask(mask)

        mask , _ , all_mask = mask_generater(face_mask,init_image)
        blurred_mask = blur_mask(mask)
        depth_image = get_depth_map(init_image)
        if False:
            init_image = background_worker(
            init_image = init_image,
            mask_image = mask,
            blur_strength = 15,
            tint_color = (0,0,0),
            seed = seed)
            
            init_image = add_random_spots(init_image,
            all_mask,
            num_spots=400,
            min_size=10, 
            max_size=40, 
            tint_color=(100, 100, ), 
            opacity=0.5,
            seed=seed
            )
            init_image = distort_image(init_image, 
            all_mask, 
            grid_size=30, 
            distortion_strength=25,
            seed = seed)
        elif False:
            init_image = distort_image(init_image, 
                all_mask, 
                grid_size=30, 
                distortion_strength=25,
                seed = seed)
        elif True:
            init_image = add_random_spots(init_image,
                all_mask,
                num_spots=1200,
                min_size=5, 
                max_size=15, 
                tint_color=(50, 150, 51), 
                opacity=0.5,
                seed=seed
                )
            init_image = distort_image(init_image, 
                all_mask, 
                grid_size=30, 
                distortion_strength=50,
                seed = seed)
        elif False:
            init_image = background_replace(init_image, all_mask)
            init_image = add_random_spots(init_image,
                all_mask,
                num_spots=400,
                min_size=20, 
                max_size=45, 
                tint_color=(250, 250, 250), 
                opacity=0.2,
                seed=seed
                )
            
        prompt_embeds = compel_prompt(prompt)


        prompt_embeds = compel_prompt(prompt)
        
        #ip_adapter_image = Image.open("./picture/picture.jpg")

        pipe_output = pipe(
        #ip_adapter_image=ip_adapter_image,
        prompt_embeds=prompt_embeds, 
        negative_prompt=negative_prompt,  
        #height=512, width=512,     # 图片大小，在此处不会生效
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator,
        strength=strength,
        image=init_image,
        mask_image = blurred_mask,
        control_image=depth_image,
        controlnet_conditioning_scale = 1.0
        )
        final_image = pipe_output.images[0]

        print("5")
    
    
    
    #用比较小的蒙版叠加，防止融入不该融入的东西
    #大蒙版可以制作渐变
    if face_mask["repair_face"]["repair_face_bool"] == True:
        #big_mask = blur_mask(big_mask)
        final_image = face_repairer(init_image,pipe_output.images[0], blurred_mask,face_mask["repair_face"]["repair_strength"])
        


    
    print(f"----本次测试输出：")
    print(f"\nprompt：{prompt}\nnegative_prompt：{negative_prompt}\n指导程度：{guidance_scale}\n采样步数：{steps}\n更改强度：{strength}\n种子：{seed}")
    #pipe_output.images[0].save("./result_test/A.png")
    #暂时保存
    return(final_image)


This part mainly from CLIP-Lora https://github.com/MaxZanella/CLIP-LoRA

I have changed a lot to meet my project, but the main idea is the same.

the key is apply_lora function:

    list_lora_layers = apply_lora(args, clip_model)

then clip_model has a lora


    mark_only_lora_as_trainable(clip_model)
    
    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)

when inference:

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    img_path = "/path/to/a/local/img/xxx.jpg"
    image = preprocess(Image.open(img_path)).unsqueeze(0)
    text = tokenizer(["a diagram", "a dog", "a cat"])
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)




so it is very simple to implement lora in CLIP


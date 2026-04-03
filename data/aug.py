import albumentations as albu

def get_transforms(size):
    pipeline = albu.Compose([
        albu.Resize(height=size[0], width=size[1]),
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
        albu.Perspective(scale=(0.02, 0.08), p=0.3),
        albu.OneOf([
            albu.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.4, hue=0.2, p=1.0),
            albu.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=20, p=1.0),
            albu.ChannelShuffle(p=1.0),
        ], p=0.4),
        albu.ToGray(p=0.2)
    ], additional_targets={'image1': 'image', 'image2': 'image', 'image3': 'image'}, is_check_shapes=False)

    def process(hr, lr1, lr2, lr3):
        r = pipeline(image=hr, image1=lr1, image2=lr2, image3=lr3)
        return r['image'], r['image1'], r['image2'], r['image3']

    return process

def get_transforms_fortest(size):
    pipeline = albu.Compose([
        albu.Resize(height=size[0], width=size[1])
    ], additional_targets={'image1': 'image', 'image2': 'image', 'image3': 'image'}, is_check_shapes=False)

    def process(hr, lr1, lr2, lr3):
        r = pipeline(image=hr, image1=lr1, image2=lr2, image3=lr3)
        return r['image'], r['image1'], r['image2'], r['image3']

    return process

def get_normalize():
    normalize = albu.Compose([
        albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    def process(a):
        r = normalize(image=a)
        return r['image']

    return process

import albumentations as albu

def get_transforms(size):
    pipeline = albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
        albu.Resize(height=size[0], width=size[1])
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

import albumentations as albu


def get_transforms(size):
    augs = {'weak': albu.Compose([albu.HorizontalFlip(),
                                  ]),
            'geometric': albu.OneOf([albu.HorizontalFlip(always_apply=True),
                                     albu.ShiftScaleRotate(always_apply=True),
                                     albu.Transpose(always_apply=True),
                                     albu.OpticalDistortion(always_apply=True),
                                     albu.ElasticTransform(always_apply=True),
                                     ])
            }

    aug_fn = augs['geometric']
    crop_fn = {'random': albu.RandomCrop(size[0], size[1], always_apply=True),
               'center': albu.CenterCrop(size[0], size[1], always_apply=True)}['random']

    effect = albu.OneOf([albu.MotionBlur(blur_limit=21, always_apply=True),
                         albu.RandomRain(always_apply=True),
                         albu.RandomFog(always_apply=True),
                         albu.RandomSnow(always_apply=True)])
    motion_blur = albu.MotionBlur(blur_limit=55, always_apply=True)

    resize = albu.Resize(height=size[0], width=size[1])

    # pipeline = albu.Compose([resize], additional_targets={'target': 'image'})
    pipeline = albu.Compose([resize])

    pipforblur = albu.Compose([effect])

    def process(a):
        # f = pipforblur(image=a)
        # r = pipeline(image=a, target=b)
        r = pipeline(image=a)
        # return r['image'], r['target']
        return r['image']

    return process


def get_transforms_fortest(size):
    resize = albu.Resize(height=size[0], width=size[1])

    effect = albu.OneOf([albu.MotionBlur(always_apply=True),
                         albu.RandomRain(always_apply=True),
                         albu.RandomFog(always_apply=True),
                         albu.RandomSnow(always_apply=True)])
    motion_blur = albu.MotionBlur(blur_limit=51, always_apply=True)

    pipeline = albu.Compose([resize], additional_targets={'target': 'image'})

    def process(a, b):
        r = pipeline(image=a, target=b)
        return r['image'], r['target']

    return process


def get_normalize():
    normalize = albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # normalize = albu.Compose([normalize], additional_targets={'target': 'image'})
    normalize = albu.Compose([normalize])

    def process(a):
        r = normalize(image=a)
        return r['image']

    return process


def get_extra_augments():
    """
    Prepared extra augmentations for more realistic training data (inter-frame
    misalignment, exposure variation). NOT active by default — wire in by calling
    this in the dataset and applying it per LR frame before normalization.
    """
    pipeline = albu.Compose([
        albu.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.0, rotate_limit=5,
                              border_mode=0, p=0.5),
        albu.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03, p=0.5),
    ])

    def process(a):
        return pipeline(image=a)['image']

    return process

import deeplake
ds_train = deeplake.load("hub://activeloop/tiny-imagenet-train")
ds_test = deeplake.load("hub://activeloop/tiny-imagenet-test")
ds_validation = deeplake.load("hub://activeloop/tiny-imagenet-validation")
dataloader = ds_train.pytorch(num_workers=0, batch_size=4, shuffle=False)
dataloader = ds_train.tensorflow()


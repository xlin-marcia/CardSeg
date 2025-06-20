import os
import math
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def open_nifti_image(file_path):
    img = nib.load(file_path)
    img_tensor = torch.tensor(img.get_fdata(), dtype=torch.float32)
    return img_tensor


def view_nifti_image(image_tensor, cmap="gray", save_path=None):
    assert image_tensor.dim() == 3
    H, W, C = image_tensor.shape

    ncols = math.ceil(math.sqrt(C))
    nrows = math.ceil(C / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = axes.flatten()

    for i in range(C):
        slice_2d = image_tensor[:, :, i].numpy()
        axes[i].imshow(slice_2d, cmap=cmap)
        axes[i].set_title(f"Channel {i}")
        axes[i].axis("off")

    for i in range(C, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def interpolate_depth_channels(image_tensor, target_channels=15):
    assert image_tensor.dim() == 3, "Input must be 3D tensor [H, W, C]"
    H, W, D = image_tensor.shape

    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # to make [1, 1, C, H, W]
    interpolated = F.interpolate(image_tensor, size=(target_channels, H, W), mode="trilinear", align_corners=False)
    return interpolated.squeeze(0).squeeze(0).permute(1, 2, 0)


def fix_dimensions(image_tensor, height=256, width=256):
    assert image_tensor.dim() == 3, "Input must be 3D tensor [H, W, C]"
    image_tensor = image_tensor.permute(2, 0, 1)
    C, H, W = image_tensor.shape

    if H != height or W != width:
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        ).squeeze(0)

    return image_tensor.permute(1, 2, 0)


# folder_path = "/home/soni/Downloads/patient100"
# for filename in os.listdir(folder_path):
#     if filename.endswith(".nii.gz"):
#         file_path = os.path.join(folder_path, filename)
#         img = open_nifti_image(file_path)
#         print(f"{img.shape} : {filename}")

img = open_nifti_image("/home/soni/Downloads/patient100/patient100_frame13.nii.gz")
print("Original Image Tensor Shape:", img.shape)

img = interpolate_depth_channels(img)
img = fix_dimensions(img)
print("New Image Tensor Shape:", img.shape)

view_nifti_image(img)  # , save_path="/home/soni/Downloads/patient100/image.png")

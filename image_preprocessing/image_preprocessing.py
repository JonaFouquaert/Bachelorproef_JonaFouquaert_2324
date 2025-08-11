import numpy as np
import rasterio, cv2
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from PIL import Image



def read_images(file_path_1, file_path_2, height, width):
  with rasterio.open(file_path_1) as src1, rasterio.open(file_path_2) as src2:
   
    out_shape_1 = (src1.count, height, width)
    out_shape_2 = (src2.count, height, width) 
    
  
    num_bands_to_read = 3 
    out_shape_1 = (num_bands_to_read, height, width)
    out_shape_2 = (num_bands_to_read, height, width)

    print(f"Reading {file_path_1} into shape {out_shape_1}")
    
    data_1_resampled = src1.read(
        (1, 2, 3), # Specify bands
        out_shape=out_shape_1,
        resampling=Resampling.bilinear 
    )

    print(f"Reading {file_path_2} into shape {out_shape_2}")
    data_2_resampled = src2.read(
        (1, 2, 3), # Specify bands
        out_shape=out_shape_2,
        resampling=Resampling.bilinear
    )

  image_1_resampled = np.moveaxis(data_1_resampled, 0, -1)
  image_2_resampled = np.moveaxis(data_2_resampled, 0, -1)

  return image_1_resampled, image_2_resampled

def crop_images(img_1, img_2, top, bottom, left, right):

  # Crop the region
  city_crop_1 = img_1[top:bottom, left:right]
  city_crop_2 = img_2[top:bottom, left:right]

  return city_crop_1, city_crop_2


def compute_exg(rgb_image):
    rgb = rgb_image.astype(np.float32)
    R, G, B = rgb[..., 2], rgb[..., 1], rgb[..., 0] 
    exg = 2 * G - R - B
    return exg

def threshold_exg(exg):
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary_mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_mask

# def threshold_exg(exg, thresh_value=128):
#     """
#     Apply a manual threshold to the ExG vegetation index.
    
#     Parameters:
#         exg (ndarray): Excess Green index array.
#         thresh_value (int): Threshold value (0–255). Higher means stricter vegetation detection.
        
#     Returns:
#         binary_mask (ndarray): Binary vegetation mask (uint8, 0 or 255).
#     """
#     # Normalize ExG to 0–255 for thresholding
#     exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#     # Apply manual threshold
#     _, binary_mask = cv2.threshold(exg_norm, thresh_value, 255, cv2.THRESH_BINARY)

#     return binary_mask

def clean_mask(binary_mask):
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned

def generate_vegetation_change_label(veg_mask_before, veg_mask_after):
    
    before = (veg_mask_before > 0).astype(np.uint8)
    after = (veg_mask_after > 0).astype(np.uint8)

    change = np.abs(after - before)

    return change  # shape: [H, W], values in {0, 1}

def visualise(vegetation_mask_1, vegetation_mask_2, change_label, x1, x2):
   # Visualization
   plt.figure(figsize=(30, 10))

   plt.subplot(1, 5, 1)
   plt.imshow(x1)
   plt.title("Before image")
   plt.axis("off")

   plt.subplot(1, 5, 2)
   plt.imshow(vegetation_mask_1, cmap='gray')
   plt.title("Before vegetation mask")
   plt.axis("off")

   plt.subplot(1, 5, 3)
   plt.imshow(change_label, cmap='gray')
   plt.title("change labels")
   plt.axis("off")

   plt.subplot(1, 5, 4)
   plt.imshow(vegetation_mask_2, cmap='gray')
   plt.title("After vegetation mask")
   plt.axis("off")

   plt.subplot(1, 5, 5)
   plt.imshow(x2)
   plt.title("After image")
   plt.axis("off")

   plt.tight_layout()
   plt.show()


def save_image(img_before_cropped, img_after_cropped, path_img_before, path_img_after):
   Image.fromarray(img_before_cropped).save(path_img_before)
   Image.fromarray(img_after_cropped).save(path_img_after)

def load_image(path_img_before, path_img_after):
   img_before_cropped = np.array(Image.open(path_img_before))
   img_after_cropped = np.array(Image.open(path_img_after))
   return img_before_cropped, img_after_cropped


def load_image_pairs_labels(image_paths, normalized=False):

  image_pairs = []
  labels = []

  for paths in image_paths:

    img_1, img_2 = read_images(paths[0], paths[1], paths[2], paths[3])

    city_crop_1, city_crop_2 = crop_images(img_1, img_2, paths[4], paths[5], paths[6], paths[7])

    for i in range(0, city_crop_1.shape[0], paths[8]):

      for j in range(0, city_crop_1.shape[0], paths[8]):
          
          x1 = city_crop_1[i:i+paths[8], j:j+paths[8]]
          x2 = city_crop_2[i:i+paths[8], j:j+paths[8]]
          
          # save_image(x1, x2, f"Data/Gent/Gent_2020/gent_2020_i{i}_j{j}.png", f"Data/Gent/Gent_2024/gent_2024_i{i}_j{j}.png")

          if normalized:
            x1 = x1 / 255.0
            x2 = x2 / 255.0

          exg_1 = compute_exg(x1)
          exg_2 = compute_exg(x2)

          binary_mask_1 = threshold_exg(exg_1)
          binary_mask_2 = threshold_exg(exg_2)

          veg_mask_1 = clean_mask(binary_mask_1)
          veg_mask_2 = clean_mask(binary_mask_2)

          change_labeled = generate_vegetation_change_label(veg_mask_1, veg_mask_2)

          # visualise(veg_mask_1, veg_mask_2, change_labeled, x1, x2)

          image_pairs.append((x1, x2))
          labels.append(change_labeled)

  return image_pairs, labels





import torchvision.transforms as transforms
import cv2
import torch
import numpy as np
import albumentations as A
import PIL.Image
class corruptions:
    r"""Benchmark for evaluating agents in environments using Albumentations"""

    def __init__(
            self, visual_corruption, visual_severity, im_size=256
) -> None:
        self._corruptions = visual_corruption
        self._severities = visual_severity
        self.to_pil = transforms.ToPILImage()

    def apply(self, frame):
        if isinstance(frame, PIL.Image.Image):
            im = np.array(frame)

        w_im, h_im = im.shape[1], im.shape[0]
        # define a transform
        if self._corruptions == "Jitter":
            im = self.to_pil(im)
            brightness = np.random.uniform(0.5,1.5,1)[0]
            con = np.random.uniform(0.2,0.7,1)[0]
            sat = np.random.uniform(0.6,1.4,1)[0]
            hue = np.random.uniform(0.1,0.4,1)[0]

            transform = transforms.ColorJitter(brightness=brightness, contrast=con, 
                                                saturation=sat, hue=hue)
            im = transform(im)

        elif self._corruptions == "Occlusion":
            # im = self.to_pil(im)
            w_box, h_box = 12*self._severities, 12*self._severities
            # Random box's colour
            rgb = np.random.randint(0,255,3)
            r,g,b = int(rgb[0]), int(rgb[1]), int(rgb[2])

            # Random box's coordinate
            x,y = np.random.randint(0, w_im-w_box, 1)[0], np.random.randint(0, h_im-h_box, 1)[0]

            im = occlude_with_color_box(im, box_color=(r, g, b), box_coordinates=(x, y, w_box, h_box))

        elif self._corruptions == "Glare":
            radius = self._severities * 50
            transform = A.Compose(
                [A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0.5, p=1, src_radius=radius)],
            )
            im = transform(image=im)["image"]

        elif self._corruptions == "Shadow":
            num_shadows_upper = self._severities * 1
            transform = A.Compose(
                [A.RandomShadow(num_shadows_lower=1, num_shadows_upper=num_shadows_upper, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=1)],
            )
            im = transform(image=im)["image"]

        elif self._corruptions == "Fog":
            transform = A.Compose(
                        [A.RandomFog(fog_coef_lower=0.8, fog_coef_upper=0.9, alpha_coef=0.3, p=1)],
                    )
            im = transform(image=im)["image"]

        elif self._corruptions == "Snow":
            transform = A.Compose(
                [A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=1)],
            )
            im = transform(image=im)["image"]

        elif self._corruptions == "Rain":
            transform = A.Compose(
                [A.RandomRain(brightness_coefficient=0.8, drop_width=2, blur_value=1, p=1)],
            )
            im = transform(image=im)["image"]


        elif self._corruptions == "Light Out":
            threshold = self._severities * 0.17
            black_screen = np.random.uniform(0,1,1)[0] > 0.5
            if black_screen:
                im = np.zeros_like(im)

        elif self._corruptions == "Motion Blur":
            transform = A.Compose(
                [A.augmentations.blur.transforms.MotionBlur(blur_limit=7, allow_shifted=True, always_apply=True)]
            )
            im = transform(image=im)["image"]

        # apply above transform on input image
        
        if not(isinstance(im, PIL.Image.Image)):
            im = self.to_pil(im)

        return im


def occlude_with_color_box(img, box_color, box_coordinates):
    # Copy the image to avoid modifying the original
    img_with_box = img.copy()

    # Extract box coordinates
    x, y, width, height = box_coordinates

    # Draw a filled rectangle on the image to occlude
    cv2.rectangle(img_with_box, (x, y), (x + width, y + height), box_color, thickness=-1)
    return img_with_box
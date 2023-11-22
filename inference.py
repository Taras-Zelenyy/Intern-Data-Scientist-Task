import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from metrics import dice_coef, dice_coef_loss


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = img.astype(np.float32) / 255.0
    return img[np.newaxis, ...]


def display_result(img, pred_mask):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(pred_mask, cmap="gray")
    axs[1].set_title('Predicted Mask')
    axs[1].axis('off')

    plt.show()


def inference(image_folder, model_path):
    model = load_model(model_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    images = [os.path.join(image_folder, image_name) for image_name in os.listdir(image_folder)]

    for image_path in images:
        img = preprocess_image(image_path)
        pred_mask = model.predict(img).squeeze()
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        img = img.squeeze()
        display_result(img, pred_mask)


if __name__ == "__main__":
    """ 
        Place the images for manual testing of the model in the 'manual_test_images' folder and run this script.
        
        If you want to use a folder from the system, provide the full path to the folder with the test images.
        Example: C:/.../test_image_folder
    """
    TEST_IMAGE_FOLDER = "./manual_test_images"
    MODEL_PATH = "trained_model/optimized_final_model.h5"

    # Інференс для зображень з тестового датасету
    inference(TEST_IMAGE_FOLDER, MODEL_PATH)

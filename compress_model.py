import tensorflow as tf
from metrics import dice_coef, dice_coef_loss


"""
    Loading an existing Keras model.
    It is important to note that in order to correctly load a model that uses custom losses or metrics,
    we must explicitly pass these functions as `custom_objects`.
"""
model = tf.keras.models.load_model('./trained_model/final_model.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

"""
    Remove unused weights from the model before saving.
    This helps to reduce the size of the model file because the weights associated with the optimizer
    that are not needed for inference (using the model for prediction) are removed.
    This is useful in the context of a file size limit, for example, when uploading models to GitHub.
"""
model.save('./trained_model/optimized_final_model.h5', include_optimizer=False)


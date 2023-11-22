import tensorflow.keras.backend as K


def dice_coef(y_true, y_pred, smooth=100):
    """ Function for calculating the Dice coefficient """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred):
    """ Function for calculating losses by the Dice coefficient """
    return 1 - dice_coef(y_true, y_pred)

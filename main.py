import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from unet import create_unet_model
from airbus_dataset import DataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from metrics import dice_coef, dice_coef_loss


def main():
    """ Define paths to data """
    '''
        If you encounter an error like this one: 
        Unable to load image at path XXXXXXXXX.jpg, 
        please set the full path to the image directories
        and the error should disappear. 
        Example: C:/Project name/.../image_folder
    '''
    # image_folder = "path/train_image"  # Path to the folder with images
    # csv_file = 'path/train_ship_segmentations_v2.csv'  # Path to the csv file with data

    image_folder = "./first_100_images"
    csv_file = './first_100_data.csv'  # Path to the csv file with data

    """ Load data """
    all_data = pd.read_csv(csv_file)
    all_data['WithShip'] = ~all_data['EncodedPixels'].isna()

    """ Balance data """
    ships_count = all_data['WithShip'].sum()
    ships_df = all_data[all_data['WithShip']]
    no_ships_df = all_data[~all_data['WithShip']].sample(n=ships_count // 2, random_state=42)
    balanced_data = pd.concat([ships_df, no_ships_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    """ Split data """
    train_df, validation_df = train_test_split(balanced_data, test_size=0.2, random_state=42, stratify=balanced_data['WithShip'])

    """ Create data generators """
    batch_size = 12
    train_generator = DataGenerator(train_df, image_folder, batch_size=batch_size, augment=True)
    val_generator = DataGenerator(validation_df, image_folder, batch_size=batch_size, augment=False)

    """ Create model """
    model = create_unet_model(input_shape=(512, 512, 3), num_classes=1)
    optimizer = Adam(learning_rate=0.03)
    model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])

    """ Callbacks for saving model"""
    callbacks = [
        ModelCheckpoint('model_epoch_{epoch:02d}.h5', save_best_only=False),
        ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    ]

    """ Train model """
    history = model.fit(train_generator, epochs=4, validation_data=val_generator, callbacks=callbacks)

    """ Save model """
    model.save('final_model.h5')


if __name__ == "__main__":
    main()

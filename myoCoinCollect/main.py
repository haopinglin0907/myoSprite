from myorawfast import MyoMain
from tensorflow.python.keras.layers import deserialize
from tensorflow.python.keras.saving import saving_utils
from pickle import load
import pandas as pd


# need this function for reading the pretrained model in pickle format
def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

def emg_buffer_handler(emg_buffer, *args):
    print(None)

if __name__ == "__main__":
    ID = input('Please enter subject ID: ')
    # load pretrained model that was trained on Snow, Dan, Rita, and Hao-Ping (2 sessions each person)
    # Data augmentation was applied to tackle the rotation / orientation issue
    model = load(open(f'{ID}/model.pkl', 'rb'))
    gestureProfile = pd.read_pickle(f'{ID}/gestureProfile_{ID}.pkl')
    gestureProfile = gestureProfile.set_index('Gesture')
    
    mm = MyoMain(model, gestureProfile)
    # mm.add_emg_buffer_handler(emg_buffer_handler)

    mm.connect()
    mm.no_sleep()
    mm.start_collect()
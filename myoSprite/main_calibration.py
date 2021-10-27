from myorawfast_calibration import MyoMain

def emg_buffer_handler(emg_buffer, *args):
    print(None)

if __name__ == "__main__":

    mm = MyoMain()
    # mm.add_emg_buffer_handler(emg_buffer_handler)

    mm.connect()
    mm.no_sleep()
    mm.start_collect()
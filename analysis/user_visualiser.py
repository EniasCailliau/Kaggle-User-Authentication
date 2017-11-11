import matplotlib.pyplot as plt
import pandas as pd

from feature_extraction import data_loader
from feature_extraction._data import sensor


def visualizeinterval_data(intervals_flat, i):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    hand_accelero = pd.DataFrame(
        {'X': intervals_flat.iloc[i]["interval_data"][sensor.hand_accelerometer_X_axis],
         'Y': intervals_flat.iloc[i]["interval_data"][sensor.hand_accelerometer_Y_axis],
         'Z': intervals_flat.iloc[i]["interval_data"][sensor.hand_accelerometer_Z_axis]})

    hand_gyro = pd.DataFrame(
        {'X': intervals_flat.iloc[i]["interval_data"][sensor.hand_gyroscope_X_axis],
         'Y': intervals_flat.iloc[i]["interval_data"][sensor.hand_gyroscope_Y_axis],
         'Z': intervals_flat.iloc[i]["interval_data"][sensor.hand_gyroscope_Z_axis]})

    chest_accelero = pd.DataFrame(
        {'X': intervals_flat.iloc[i]["interval_data"][sensor.chest_accelerometer_X_axis],
         'Y': intervals_flat.iloc[i]["interval_data"][sensor.chest_accelerometer_Y_axis],
         'Z': intervals_flat.iloc[i]["interval_data"][sensor.chest_accelerometer_Z_axis]})

    chest_gyro = pd.DataFrame(
        {'X': intervals_flat.iloc[i]["interval_data"][sensor.chest_gyroscope_X_axis],
         'Y': intervals_flat.iloc[i]["interval_data"][sensor.chest_gyroscope_Y_axis],
         'Z': intervals_flat.iloc[i]["interval_data"][sensor.chest_gyroscope_Z_axis]})

    plot1 = hand_accelero.plot(ax=axes[0, 0], title="hand accelerometer interval_data")
    plot1.set_xlabel("time")
    plot1.set_ylabel("acceleration")

    plot2 = hand_gyro.plot(ax=axes[0, 1], title="hand gyroscope interval_data")
    plot2.set_xlabel("time")
    plot2.set_ylabel("acceleration")

    plot3 = chest_accelero.plot(ax=axes[1, 0], title="chest gyroscope interval_data")
    plot3.set_xlabel("time")
    plot3.set_ylabel("acceleration")

    plot4 = chest_gyro.plot(ax=axes[1, 1], title="chest gyroscope interval_data")
    plot4.set_xlabel("time")
    plot4.set_ylabel("acceleration")

    title = "Subject {} Activity {}".format(intervals_flat.iloc[i]["subject"], intervals_flat.iloc[i]["activity"])
    st = fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    plt.savefig("vis/" + title + '.png')


if __name__ == '__main__':
    train_flat, test_flat = data_loader.create_flat_intervals_structure()
    for i in range(0, 3000, 100):
        visualizeinterval_data(train_flat, i)
        print(i)

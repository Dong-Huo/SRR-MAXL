import openpyxl
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

white_excel = "datasets/MNWHL4_Data.xlsx"
amber_excel = "datasets/M595L4_data.xlsx"

sunlight_txt = "datasets/sun_light.txt"


def load(LED="white"):
    if LED == "white":
        filename = white_excel
    else:
        filename = amber_excel
    wb_obj = openpyxl.load_workbook(filename)

    # Read the active sheet:
    sheet = wb_obj.active

    wavelength = []
    intensity = []

    for row in sheet.iter_rows(min_row=3):
        wavelength.append(row[2].value)
        intensity.append(row[3].value)

    # plt.plot(wavelength, intensity, '--')
    # plt.legend(['data'], loc='best')
    # plt.show()

    return wavelength, intensity


def load_sunlight():
    filename = sunlight_txt
    wavelength = []
    intensity = []

    with open(filename, 'r', encoding="UTF-8") as source:
        lines = source.readlines()

        for line in lines:
            line = line.strip().split("	")
            wavelength.append(float(line[0]))
            intensity.append(float(line[1]))

    return wavelength, intensity


def interpolate(wavelength, intensity, new_wavelength=range(400, 710, 10)):
    f = interp1d(wavelength, intensity)
    new_intensity = f(new_wavelength)

    return new_intensity

# wavelength, intensity = load_sunlight()
#
# new_intensity = interpolate(wavelength, intensity, range(400, 730, 10))
# plt.plot(wavelength, intensity, '--', range(400, 730, 10), new_intensity, 'o')
# plt.legend(['old', "sampling"], loc='best')
# plt.show()

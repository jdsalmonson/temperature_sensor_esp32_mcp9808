## temperature_sensor_esp32_mcp9808

This is the repository for data and analysis for my blog post [Quantifying the Thermal Benefits of Replacement of my House's Front Door](https://jdsalmonson.github.io/livermore-summer-2024/).

Highlights of key directories or files of interest are

- [esp32_wifi](esp32_wifi) - the build directory for the ESP-IDF code project for the ESP32 microcontroller-based temperature sensors.
  - [README.md](esp32_wifi/README.md) - a rudimentary README file for the microcontroller.
  - [main.c](esp32_wifi/main/main.c) - key source file for microcontroller.
- [temperature_logger.py](scripts/temperature_logger.py) - Python script to query ESP32 temperature sensors and log data.
- [horemheb](horemheb/horemheb) - Python package (named after an [Egyptian pharoah](https://en.wikipedia.org/wiki/Horemheb)) of tools for analysis and plots produced in this blog.
- [100_400_plot_temperature.ipynb](notebooks/100_400_plot_temperature.ipynb) - plot raw temperature data.
- [200_200_analyze_temperature_over_params.ipynb](notebooks/200_200_analyze_temperature_over_params.ipynb) - Analysis of data, application of Newton's Law of Cooling.
- [300_200_optimize_cooling_de.ipynb](notebooks/300_200_optimize_cooling_de.ipynb) - Analysis of dynamical cooling model.
- [100_400_analyze_LVK_weather_data.ipynb](notebooks/100_400_analyze_LVK_weather_data.ipynb) - For reference, load and look at the temperature and humidity measurements from the LVK airport over the duration in question here.  I didn't apply this data to the analysis, but it is available.  It is possible that there is a correlation between cooling and wind speed or direction.

---

1/22/2025

An ESP32S3 board reads temperature information from an MCP9808 temperature sensor via I2C.


---

The set up of this project follows that of `labrador_classifier`.

Some basic setup:
```bash
micromamba env create -p "./.venv_temp_sense" "python=3.13"
micromamba activate "./.venv_temp_sense"
micromamba install "uv"
uv pip install zeroconf requests # for web access and mDNS hostname lookup
uv pip install rich
uv pip install ipykernel ipywidgets # for notebooks
uv pip install pandas matplotlib
uv pip install scipy
```

---
###

From prompts, I created a 3.13 python environment with `idf_tools.py install-python-env`.  

Then, as per the "Getting started with the ESP-IDF" Evernote, the following commands created a VScode session that did create a `build/` directory:
```bash
. ~/stash/esp/esp-idf/export.sh
export IDF_XTENSA_GCC="$(which xtensa-esp32-elf-g++)"  # <- not sure if this is crucial, but it works
cd ~/Work/Animata/Esp_projects/temperature_sensor_esp32_mcp9808/esp32
cursor .
```


---
### Set up package

Created a package to manage the loading and fitting of this data.  Named package after Egyptian pharoah Horemheb:

```
cd horemheb
uv pip install -e .
```

---

March 20, 2025

Downloaded humidity and wind data over this period of time for LVK from the [National Weather Service](https://www.weather.gov/wrh/timeseries?site=klvk).  Click `Advanced Options` and in the pop-up window click `Permanent Chart` to select the item to be plotted and `Gather Historical Data` to select a date range.  Then, from the resulting plot, click the bars in the upper right and the click `Download CSV` from the pull-down menu.


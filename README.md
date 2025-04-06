## temperature_sensor_esp32_mcp9808

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


---
### Junk DNA:

It is an interesting idea to define a package (or packages) for the project and to be liberated by using pseudo-random names.  I am using Ancient Egyptian Pharoah names as pseudo-random names for packages.  In this case I create a `narmer/narmer/__init__.py` directory structure with `__init__.py` containing placeholder docstring and version.  Then craate the package with `flit`:
```bash
cd ./narmer
flit init
```
which creates `./narmer/pyproject.toml`.  I added a few dependencies, following the blog's example.  Then I install the `dev` version of the package (again from with `./narmer`):
```bash
uv pip install -e '.[dev]'
```

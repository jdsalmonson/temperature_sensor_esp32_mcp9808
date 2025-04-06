Use the example build for the temperature sensor at: `~/stash/esp-idf-lib/examples/mcp9808/default`.

Also, see `~/stash/tf-fomo/esp32/fomo_camera_to_servos` as an example.

Note that I'm using mDNS to translate the EPS32 hostnames into IP addresses.  I needed to install this, for ESP-IDF v5.0+, with the line
`idf.py add-dependency espressif/mdns`
and then add mDNS stuff in `idf.py menuconfig` and add `mdns` to the `REQUIRES` list in `main/CMakeLists.txt`.  Then I could include `<mdsn.h>` and necessary code to `main/main.c`.  Then `scripts/temperature_logger.py` uses the `zeroconf` library to access the names, look up the IP address and then poll the temperatures.

---
To configure:
```bash
. ~/stash/esp/esp-idf/export.sh
export IDF_XTENSA_GCC="$(which xtensa-esp32-elf-g++)"  # <- not sure if this is crucial, but it works for building in cursor VScode
cd ~/Work/Animata/Esp_projects/temperature_sensor_esp32_mcp9808/esp32
cursor .  # will automatically build successfully
```

To build:
```bash
idf.py set-target esp32s3
idf.py build
```

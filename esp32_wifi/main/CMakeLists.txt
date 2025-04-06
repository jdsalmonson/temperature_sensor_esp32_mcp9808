idf_component_register(SRCS "main.c"
                    INCLUDE_DIRS "."
                    REQUIRES driver esp_wifi esp_http_server nvs_flash mdns
                    )

#include <stdio.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <esp_system.h>
#include <string.h>
#include <esp_err.h>
#include <esp_log.h>
#include <driver/i2c.h>
#include <esp_wifi.h>
#include <esp_event.h>
#include <esp_http_server.h>
#include <esp_wifi.h> // for wifi strength
#include <nvs_flash.h>
#include <esp_netif.h>
#include <mdns.h>

#define I2C_MASTER_SDA 5
#define I2C_MASTER_SCL 6
#define I2C_MASTER_FREQ_HZ 100000
#define I2C_PORT 0
#define MCP9808_ADDR 0x18

// WiFi credentials
#define WIFI_SSID      "JnLNet"
#define WIFI_PASS      "*****"

// MCP9808 registers
#define REG_CONFIG 0x01
#define REG_TEMP 0x05
#define REG_MANUF_ID 0x06
#define REG_DEVICE_ID 0x07

static const char *TAG = "MCP9808_WIFI";
static float current_temperature = 0.0;
static httpd_handle_t server = NULL;

static esp_err_t i2c_master_init(void)
{
    ESP_LOGI(TAG, "Configuring I2C");
    
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA,
        .scl_io_num = I2C_MASTER_SCL,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
        .clk_flags = 0
    };
    
    esp_err_t err = i2c_param_config(I2C_PORT, &conf);
    if (err != ESP_OK) return err;
    
    return i2c_driver_install(I2C_PORT, I2C_MODE_MASTER, 0, 0, 0);
}

static esp_err_t read_register(uint8_t reg, uint16_t *data)
{
    uint8_t buf[2];
    
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (MCP9808_ADDR << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg, true);
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (MCP9808_ADDR << 1) | I2C_MASTER_READ, true);
    i2c_master_read(cmd, buf, 2, I2C_MASTER_LAST_NACK);
    i2c_master_stop(cmd);
    
    esp_err_t ret = i2c_master_cmd_begin(I2C_PORT, cmd, 1000 / portTICK_PERIOD_MS);
    i2c_cmd_link_delete(cmd);
    
    if (ret == ESP_OK) {
        *data = (buf[0] << 8) | buf[1];
    }
    
    return ret;
}

static esp_err_t read_temperature(float *temp)
{
    uint16_t raw_temp;
    esp_err_t ret = read_register(REG_TEMP, &raw_temp);
    
    if (ret == ESP_OK) {
        // Convert raw temperature to celsius
        // Clear flag bits
        raw_temp &= 0x1FFF;
        // If negative, convert from 13-bit signed to 16-bit signed
        if (raw_temp & 0x1000)
            raw_temp |= 0xE000;
        // Temperature resolution is 0.0625°C
        *temp = raw_temp * 0.0625;
    }
    
    return ret;
}

// HTTP server handler
static esp_err_t temperature_handler(httpd_req_t *req)
{
    // return the temperature as a string:
    //char resp[100];
    //sprintf(resp, "Temperature: %.2f °C\n", current_temperature);

    // return the temperature as a float:
    char resp[32];
    snprintf(resp, sizeof(resp), "%.2f", current_temperature);  // Just the number

    httpd_resp_send(req, resp, strlen(resp));
    return ESP_OK;
}

// RSSI handler
static esp_err_t rssi_handler(httpd_req_t *req)
{
    wifi_ap_record_t ap_info;
    esp_err_t err = esp_wifi_sta_get_ap_info(&ap_info);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to get RSSI: %s", esp_err_to_name(err));
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Failed to get RSSI");
        return ESP_FAIL;
    }

    char resp[32];
    snprintf(resp, sizeof(resp), "%d", ap_info.rssi);
    httpd_resp_set_type(req, "text/plain");
    httpd_resp_send(req, resp, strlen(resp));
    return ESP_OK;
}

// URI handlers configuration
static const httpd_uri_t temperature_uri = {
    .uri       = "/temperature",
    .method    = HTTP_GET,
    .handler   = temperature_handler,
    .user_ctx  = NULL
};

static const httpd_uri_t rssi_uri = {
    .uri       = "/rssi",
    .method    = HTTP_GET,
    .handler   = rssi_handler,
    .user_ctx  = NULL
};

/*
// Configure HTTP server
static httpd_uri_t temperature = {
    .uri       = "/temperature",
    .method    = HTTP_GET,
    .handler   = temperature_handler,
    .user_ctx  = NULL
};

static esp_err_t rssi_get_handler(httpd_req_t *req)
{
    wifi_ap_record_t ap_info;
    esp_err_t err = esp_wifi_sta_get_ap_info(&ap_info);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to get RSSI: %s", esp_err_to_name(err));
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Failed to get RSSI");
        return ESP_FAIL;
    }

    char rssi_str[8];
    snprintf(rssi_str, sizeof(rssi_str), "%d", ap_info.rssi);
    httpd_resp_set_type(req, "text/plain");
    httpd_resp_send(req, rssi_str, strlen(rssi_str));

    return ESP_OK;
}

static const httpd_uri_t rssi = {
    .uri       = "/rssi",
    .method    = HTTP_GET,
    .handler   = rssi_get_handler,
    .user_ctx  = NULL
};

// Start HTTP server
static void start_webserver(void)
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_register_uri_handler(server, &temperature);
        //httpd_register_uri_handler(server, &rssi);
        ESP_LOGI(TAG, "Web server started");
    }
}
*/

// Start HTTP server
static void start_webserver(void)
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    
    // Increase stack size if needed
    config.stack_size = 8192;
    
    // Increase max URI handlers
    config.max_uri_handlers = 4;
    
    ESP_LOGI(TAG, "Starting server on port: '%d'", config.server_port);
    if (httpd_start(&server, &config) == ESP_OK) {
        ESP_LOGI(TAG, "Registering URI handlers");
        ESP_ERROR_CHECK(httpd_register_uri_handler(server, &temperature_uri));
        ESP_ERROR_CHECK(httpd_register_uri_handler(server, &rssi_uri));
        ESP_LOGI(TAG, "Web server started");
    } else {
        ESP_LOGI(TAG, "Error starting server!");
        return;
    }
}

// WiFi event handler
static void wifi_event_handler(void* arg, esp_event_base_t event_base,
                             int32_t event_id, void* event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        ESP_LOGI(TAG, "Connecting to WiFi...");
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGI(TAG, "Disconnected from WiFi. Retrying...");
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        start_webserver();
    }
}

// Initialize WiFi
static void wifi_init(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASS,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_LOGI(TAG, "Starting WiFi");
    ESP_ERROR_CHECK(esp_wifi_start());
}

// Initialize mDNS
static void init_mdns(void)
{
    ESP_ERROR_CHECK(mdns_init());
    
    // Set hostname (must be unique for each device)
    ESP_ERROR_CHECK(mdns_hostname_set("temp-sensor-1"));  // Or "temp-sensor-2" for second device
    // Set default instance
    ESP_ERROR_CHECK(mdns_instance_name_set("Temperature Sensor 1"));
    /*
    // Set hostname #2:
    ESP_ERROR_CHECK(mdns_hostname_set("temp-sensor-2"));  // Or "temp-sensor-2" for second device
    ESP_ERROR_CHECK(mdns_instance_name_set("Temperature Sensor 2"));
    */
    // Add service
    ESP_ERROR_CHECK(mdns_service_add(NULL, "_temperature", "_tcp", 80, NULL, 0));
    
    ESP_LOGI(TAG, "mDNS service started");
}

// Temperature sensor task
void sensor_task(void *pvParameters)
{
    ESP_LOGI(TAG, "Starting sensor task");
    
    // Initialize I2C
    ESP_ERROR_CHECK(i2c_master_init());
    ESP_LOGI(TAG, "I2C initialized");
    
    // Read and verify manufacturer ID
    uint16_t manuf_id;
    ESP_ERROR_CHECK(read_register(REG_MANUF_ID, &manuf_id));
    ESP_LOGI(TAG, "Manufacturer ID: 0x%04X", manuf_id);
    
    if (manuf_id != 0x0054) {
        ESP_LOGE(TAG, "Unexpected manufacturer ID!");
        vTaskDelete(NULL);
        return;
    }
    
    // Main loop
    while (1) {
        if (read_temperature(&current_temperature) == ESP_OK) {
            ESP_LOGI(TAG, "Temperature: %.2f °C", current_temperature);
        } else {
            ESP_LOGE(TAG, "Error reading temperature");
        }
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void app_main(void)
{
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    ESP_LOGI(TAG, "NVS initialized");

    // Start WiFi
    wifi_init();
    
    // Initialize mDNS
    init_mdns();
    
    // Create sensor task
    xTaskCreatePinnedToCore(sensor_task, "sensor_task", 4096, NULL, 5, NULL, APP_CPU_NUM);
}
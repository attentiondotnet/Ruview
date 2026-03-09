/**
 * RuView CSI Node Firmware
 * ========================
 * ESP32 firmware that captures Channel State Information (CSI) from received
 * WiFi packets and streams it to the host over UDP (and optionally over serial).
 *
 * Hardware requirements
 * ---------------------
 * - Any ESP32 board (tested on ESP32-DevKitC and ESP32-WROOM-32)
 * - Arduino core for ESP32 >= 2.0.0  (https://github.com/espressif/arduino-esp32)
 *
 * Configuration
 * -------------
 * Edit the constants in the "USER CONFIGURATION" section below before flashing.
 *
 * Wire protocol (UDP)
 * -------------------
 * Each datagram has the following layout (little-endian):
 *
 *   [ node_id   : 8 bytes  (null-padded ASCII)         ]
 *   [ timestamp : 8 bytes  (double — millis() / 1000)  ]
 *   [ rssi      : 1 byte   (int8)                      ]
 *   [ channel   : 1 byte   (uint8)                     ]
 *   [ n_sub     : 2 bytes  (uint16)                    ]
 *   [ csi_buf   : 2*n_sub  (int8 pairs: imag, real)    ]
 *
 * Serial protocol (optional)
 * --------------------------
 *   [ 0xAA       : 1 byte  (sync byte)           ]
 *   [ length     : 2 bytes (uint16 LE)            ]
 *   [ csi_buf    : length bytes (int8 pairs)      ]
 *   [ checksum   : 1 byte  (XOR of csi_buf)       ]
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <esp_wifi.h>
#include <esp_wifi_types.h>

// =============================================================================
// USER CONFIGURATION — edit before flashing
// =============================================================================

static const char *WIFI_SSID     = "YOUR_SSID";
static const char *WIFI_PASSWORD = "YOUR_PASSWORD";

// Unique identifier for this sensor node (max 7 chars + null)
static const char *NODE_ID = "node-01";

// Host IP and port where the Python RuView engine is listening
static const char *HOST_IP   = "192.168.1.100";
static const uint16_t HOST_PORT = 5005;

// WiFi channel to monitor (1–13 for 2.4 GHz)
static const uint8_t WIFI_CHANNEL = 6;

// Enable serial output in addition to UDP
static const bool SERIAL_ENABLED = true;
static const uint32_t SERIAL_BAUD = 921600;

// =============================================================================
// INTERNALS — do not edit unless you know what you're doing
// =============================================================================

#define MAX_CSI_SUBCARRIERS 128  // ESP32 can report up to 128 in HT40 mode
#define UDP_HEADER_SIZE     20   // 8 + 8 + 1 + 1 + 2
#define UDP_BUF_SIZE        (UDP_HEADER_SIZE + MAX_CSI_SUBCARRIERS * 2 + 4)

static WiFiUDP udp;
static uint8_t udpBuf[UDP_BUF_SIZE];

// ---------------------------------------------------------------------------
// CSI callback — called by the ESP-IDF WiFi driver for every received packet
// ---------------------------------------------------------------------------
static void IRAM_ATTR csi_callback(void *ctx, wifi_csi_info_t *info) {
  if (info == nullptr || info->buf == nullptr || info->len == 0) return;

  const uint16_t n_sub = (uint16_t)(info->len / 2);  // pairs of int8
  const uint32_t buf_data_len = (uint32_t)info->len;

  // ── Build UDP packet ────────────────────────────────────────────────────
  uint32_t offset = 0;

  // node_id (8 bytes, null-padded)
  memset(udpBuf + offset, 0, 8);
  strncpy((char *)(udpBuf + offset), NODE_ID, 7);
  offset += 8;

  // timestamp as double (millis / 1000)
  double ts = (double)millis() / 1000.0;
  memcpy(udpBuf + offset, &ts, sizeof(double));
  offset += sizeof(double);

  // rssi (int8)
  udpBuf[offset++] = (uint8_t)(int8_t)info->rx_ctrl.rssi;

  // channel (uint8)
  udpBuf[offset++] = WIFI_CHANNEL;

  // n_sub (uint16 LE)
  memcpy(udpBuf + offset, &n_sub, sizeof(uint16_t));
  offset += sizeof(uint16_t);

  // csi_buf (int8 pairs)
  const uint32_t copy_len = (buf_data_len < (uint32_t)(MAX_CSI_SUBCARRIERS * 2))
                             ? buf_data_len
                             : (uint32_t)(MAX_CSI_SUBCARRIERS * 2);
  memcpy(udpBuf + offset, info->buf, copy_len);
  offset += copy_len;

  // ── Send UDP ─────────────────────────────────────────────────────────────
  udp.beginPacket(HOST_IP, HOST_PORT);
  udp.write(udpBuf, offset);
  udp.endPacket();

  // ── Optional serial output ───────────────────────────────────────────────
  if (SERIAL_ENABLED) {
    const uint16_t payload_len = (uint16_t)copy_len;
    const uint8_t SYNC = 0xAA;

    // Checksum: XOR of all payload bytes
    uint8_t cs = 0;
    for (uint32_t i = 0; i < copy_len; i++) {
      cs ^= info->buf[i];
    }

    Serial.write(SYNC);
    Serial.write((uint8_t)(payload_len & 0xFF));
    Serial.write((uint8_t)(payload_len >> 8));
    Serial.write(info->buf, copy_len);
    Serial.write(cs);
  }
}

// ---------------------------------------------------------------------------
// Arduino setup
// ---------------------------------------------------------------------------
void setup() {
  if (SERIAL_ENABLED) {
    Serial.begin(SERIAL_BAUD);
    Serial.println("[RuView] CSI Node starting…");
  }

  // Connect to WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  const uint32_t connect_timeout_ms = 15000;
  const uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - start > connect_timeout_ms) {
      if (SERIAL_ENABLED) Serial.println("[RuView] WiFi connection timed out — restarting");
      ESP.restart();
    }
    delay(250);
  }

  if (SERIAL_ENABLED) {
    Serial.print("[RuView] Connected — IP: ");
    Serial.println(WiFi.localIP());
  }

  // Start UDP
  udp.begin(HOST_PORT);

  // Configure ESP32 CSI
  wifi_csi_config_t csi_cfg = {};
  csi_cfg.lltf_en           = true;   // Long training field (most informative)
  csi_cfg.htltf_en          = false;
  csi_cfg.stbc_htltf2_en    = false;
  csi_cfg.ltf_merge_en      = true;
  csi_cfg.channel_filter_en = false;  // raw CSI without channel filtering
  csi_cfg.manu_scale        = false;

  ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_cfg));
  ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(csi_callback, nullptr));
  ESP_ERROR_CHECK(esp_wifi_set_csi(true));

  if (SERIAL_ENABLED) {
    Serial.printf("[RuView] CSI enabled on channel %d — streaming to %s:%d\n",
                  WIFI_CHANNEL, HOST_IP, HOST_PORT);
  }
}

// ---------------------------------------------------------------------------
// Arduino loop — the real work is done in the CSI callback (IRAM_ATTR)
// ---------------------------------------------------------------------------
void loop() {
  // Keep the WiFi stack alive; nothing else needed here.
  delay(1000);

  if (WiFi.status() != WL_CONNECTED) {
    if (SERIAL_ENABLED) Serial.println("[RuView] WiFi lost — reconnecting…");
    WiFi.reconnect();
  }
}

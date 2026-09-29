/*
 * SkyGuard AI - ESP32 Station (WiFi HTTP Web Server + Physics QC + OLED)
 * HARDWARE: ESP32 DevKit 30-PIN + DHT11 + BMP180 + Rain sensor LM393
 *           + 1.3" I2C OLED SH1106 128x64 (SDA->GPIO21, SCL->GPIO22)
 *
 * LIBRARIES (Arduino Library Manager):
 *   1. "DHT sensor library" by Adafruit
 *   2. "Adafruit Unified Sensor" by Adafruit
 *   3. "Adafruit BMP085 Library" by Adafruit
 *   4. "ArduinoJson" by Benoit Blanchon  (v6.x)
 *   5. "U8g2" by oliver (olikraus)
 *
 * FIXES vs original paste:
 *   1. StaticJsonDocument 512->1024 (was silently dropping fields; dashboard showed "-")
 *   2. drawOLED() rewritten: clean 5-row layout, no text overlap, marquee scroll
 *      for status line when text is wider than 128px
 */

#include <math.h>
#include <Wire.h>
#include <WiFi.h>
#include <WebServer.h>
#include <DHT.h>
#include <Adafruit_BMP085.h>
#include <ArduinoJson.h>
#include <U8g2lib.h>

// ---- WiFi (DHCP) --------------------------------------------------------
const char* WIFI_SSID = "S";
const char* WIFI_PASS = "12121212";
WebServer server(80);

// ---- Station identity ---------------------------------------------------
const char* STATION_ID = "AWS-EDGE-01";

// ---- DHT11 --------------------------------------------------------------
#define DHTPIN  4
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// ---- BMP180 (I2C: SDA->GPIO21, SCL->GPIO22) ----------------------------
Adafruit_BMP085 bmp;
bool bmp_ready = false;

// ---- Rain sensor --------------------------------------------------------
#define RAIN_DIGITAL_PIN         27
#define RAIN_ANALOG_PIN          34
#define RAIN_DRY_RAW           4095
#define RAIN_WET_RAW           1000
#define RAIN_DETECT_THRESHOLD_PCT 30

// ---- OLED (1.3" SH1106, 128x64, HW I2C, shared bus with BMP180) --------
// If your panel is SSD1306-based swap this line for the SSD1306 constructor.
U8G2_SH1106_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);
// U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);

// ---- Physics constants --------------------------------------------------
const float KELVIN_OFFSET       = 273.15f;
const float TETENS_A            = 6.112f;
const float TETENS_B            = 17.67f;
const float TETENS_C            = 243.5f;
const float EPSILON_A           = 0.622f;
const float REFRACTIVE_K1       = 7.76e-5f;
const float REFRACTIVE_K2       = 3.73e-1f;
const float SCALE_TV            = 5.0f;
const float SCALE_N             = 2.0e-5f;
const float SCALE_E             = 4.0f;
const float EDGE_FLAG_THRESHOLD = 0.30f;
const float ATTR_SCALE_T        = 5.0f;
const float ATTR_SCALE_RH       = 20.0f;
const float ATTR_SCALE_P        = 5.0f;

// ========================================================================
// STRUCT DEFINITIONS  (must all appear before any function that uses them)
// ========================================================================

struct Reading {
  float t_c;
  float p_hpa;
  float rh_pct;
  bool  valid;
};

struct EquilibriumResult {
  float r_tv;
  float r_n;
  float r_e;
  float composite;
  bool  dalton_violation;
  bool  flagged;
};

struct DisplayState {
  float t_c;
  float rh_pct;
  float p_hpa;
  bool  rain_detected;
  int   rain_pct;
  bool  flagged;
  bool  dalton_violation;
  char  anomaly_var[12];  // "TEMP" / "HUMIDITY" / "PRESSURE" / "MULTI"
  float composite;
};

// ---- Global state -------------------------------------------------------
Reading     last_good  = { 0.0f, 0.0f, 0.0f, false };
DisplayState disp      = { 0, 0, 0, false, 0, false, false, "", 0 };
String      currentPayload      = "{}";
unsigned long lastSensorReadTime = 0;
bool        first_reading_done  = false;

const unsigned long SENSOR_READ_INTERVAL = 3000;

// ========================================================================
// FORWARD DECLARATIONS
// (Arduino IDE processes top-to-bottom; declare before first use)
// ========================================================================
float             saturationVaporPressureTetens(float tc);
float             actualVaporPressure(float rh, float tc);
bool              daltonsLawViolation(float e, float p);
float             virtualTemperature(float tk, float p, float rh, float tc);
float             refractiveIndex(float tk, float p, float rh, float tc);
EquilibriumResult evaluateEquilibrium(float tc, float p, float rh, Reading ref);
void              classifyAnomalySource(float tc, float rh, float p, Reading ref, char* out);
void              buildPayload(float tc, float p, float rh, bool rain_det,
                               bool rain_dig, int rain_adc, int rain_pct,
                               EquilibriumResult res);
void              handleData();
void              handleRoot();
void              handleNotFound();
void              drawOLED();
void              readSensorsAndEvaluate();

// ========================================================================
// PHYSICS HELPER FUNCTIONS
// ========================================================================

float saturationVaporPressureTetens(float tc) {
  return TETENS_A * expf((TETENS_B * tc) / (tc + TETENS_C));
}

float actualVaporPressure(float rh, float tc) {
  return (rh / 100.0f) * saturationVaporPressureTetens(tc);
}

bool daltonsLawViolation(float e, float p) {
  return e >= p;
}

float virtualTemperature(float tk, float p, float rh, float tc) {
  float e     = actualVaporPressure(rh, tc);
  float ratio = (e / p) * (1.0f - EPSILON_A);
  return tk / fmaxf(0.01f, 1.0f - ratio);
}

float refractiveIndex(float tk, float p, float rh, float tc) {
  float e = actualVaporPressure(rh, tc);
  return 1.0f + REFRACTIVE_K1 * (p / tk)
              + REFRACTIVE_K2 * (e / (tk * tk));
}

EquilibriumResult evaluateEquilibrium(float tc, float p, float rh, Reading ref) {
  float tko = tc      + KELVIN_OFFSET;
  float tkr = ref.t_c + KELVIN_OFFSET;

  float eo  = actualVaporPressure(rh, tc);
  bool  dv  = daltonsLawViolation(eo, p);

  float rtv = fabsf(virtualTemperature(tko, p,        rh,        tc)
                  - virtualTemperature(tkr, ref.p_hpa, ref.rh_pct, ref.t_c));
  float rn  = fabsf(refractiveIndex(tko, p,        rh,        tc)
                  - refractiveIndex(tkr, ref.p_hpa, ref.rh_pct, ref.t_c));
  float re  = fabsf(eo - actualVaporPressure(ref.rh_pct, ref.t_c));

  float comp = ((rtv / SCALE_TV) + (rn / SCALE_N) + (re / SCALE_E)) / 3.0f;

  EquilibriumResult res;
  res.r_tv             = rtv;
  res.r_n              = rn;
  res.r_e              = re;
  res.composite        = comp;
  res.dalton_violation = dv;
  res.flagged          = dv || (comp >= EDGE_FLAG_THRESHOLD);
  return res;
}

void classifyAnomalySource(float tc, float rh, float p, Reading ref, char* out) {
  float dT  = fabsf(tc - ref.t_c)    / ATTR_SCALE_T;
  float dRH = fabsf(rh - ref.rh_pct) / ATTR_SCALE_RH;
  float dP  = fabsf(p  - ref.p_hpa)  / ATTR_SCALE_P;

  int active = (dT > 0.5f) + (dRH > 0.5f) + (dP > 0.5f);
  if (active >= 2)          { strcpy(out, "MULTI");    return; }
  if (dT  >= dRH && dT  >= dP) strcpy(out, "TEMP");
  else if (dRH >= dT  && dRH >= dP) strcpy(out, "HUMIDITY");
  else                          strcpy(out, "PRESSURE");
}

// ========================================================================
// JSON PAYLOAD BUILDER
// FIX: StaticJsonDocument was 512 bytes. ArduinoJson silently drops fields
// once the pool is full, so temperature_c / humidity_pct / pressure_hpa
// were all missing from the JSON -> dashboard showed "-" for everything.
// 1024 bytes fits all 17 fields comfortably.
// ========================================================================
void buildPayload(float tc, float p, float rh, bool rain_det,
                  bool rain_dig, int rain_adc, int rain_pct,
                  EquilibriumResult res) {
  StaticJsonDocument<1024> doc;   // <-- was 512

  doc["station_id"]                      = STATION_ID;
  doc["timestamp_ms"]                    = millis();
  doc["temperature_c"]                   = tc;
  doc["humidity_pct"]                    = rh;
  doc["pressure_hpa"]                    = p;
  doc["rain_detected"]                   = rain_det;
  doc["rain_digital_raw"]                = rain_dig;
  doc["rain_raw_analog"]                 = rain_adc;
  doc["rain_pct"]                        = rain_pct;
  doc["status"]                          = res.flagged ? "ANOMALOUS" : "NORMAL";
  doc["composite_physics_inconsistency"] = res.composite;
  doc["flagged"]                         = res.flagged;
  doc["dalton_violation"]                = res.dalton_violation;
  doc["r_virtual_temp_k"]                = res.r_tv;
  doc["r_vapor_pressure_hpa"]            = res.r_e;
  doc["r_refractive_index"]              = res.r_n;
  doc["anomaly_source"]                  = disp.anomaly_var;

  currentPayload = "";
  serializeJson(doc, currentPayload);
}

// ========================================================================
// HTTP HANDLERS
// ========================================================================
void handleData() {
  server.sendHeader("Access-Control-Allow-Origin",  "*");
  server.sendHeader("Access-Control-Allow-Methods", "GET");
  server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
  server.send(200, "application/json", currentPayload);
}

void handleRoot() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "text/plain",
              "SkyGuard ESP32 Station Online. GET /data for JSON.");
}

void handleNotFound() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(404, "text/plain", "Not Found");
}

// ========================================================================
// OLED - drawOLED()
// ========================================================================
//
// Layout (128 x 64 px, SH1106):
//
//  y= 8   SkyGuard            AWS-EDGE-01    font_5x7  (5px/char)
//         ----------------------------------  separator
//  y=22   T:28.5C                  H:65%    font_6x10 (6px/char)
//  y=35   P:1013.3 hPa                      font_6x10
//  y=48   Rain: DRY   (or  Rain:72%)        font_6x10
//         ----------------------------------  separator
//  y=62        >> NORMAL <<                  font_6x10, centered
//
// Status line (row 5) character counts at 6px/char:
//   ">> NORMAL <<"               12ch = 72px  -> always fits, centered
//   "!! ANOMALOUS [TEMP] !!"     22ch = 132px -> wider than display, scrolled
//   "!! ANOMALOUS [HUMIDITY] !!" 26ch = 156px -> scrolled
//   "!! ANOMALOUS [MULTI] !!"    24ch = 144px -> scrolled
//
// Scroll is right-to-left marquee at ~20 px/s. setClipWindow() prevents
// text spilling above the separator line.
//
// Rate-limited to ~30 FPS so loop() stays responsive.
// ========================================================================
void drawOLED() {
  static unsigned long lastFrameMs  = 0;
  static int           scrollX      = 128;   // start off-screen right
  static unsigned long lastScrollMs = 0;

  unsigned long now = millis();
  if (now - lastFrameMs < 33) return;        // ~30 FPS cap
  lastFrameMs = now;

  u8g2.clearBuffer();

  // ---- Boot splash (until first sensor read finishes) ------------------
  if (!first_reading_done) {
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(16, 28, "SkyGuard AI");
    u8g2.drawStr(22, 44, "Booting...");
    u8g2.sendBuffer();
    return;
  }

  // ---- Row 1: header ---------------------------------------------------
  // font_5x7: "SkyGuard" = 40px, "AWS-EDGE-01" = 55px, gap = 33px  -> OK
  u8g2.setFont(u8g2_font_5x7_tf);
  u8g2.drawStr(0, 8, "SkyGuard");
  u8g2.drawStr(127 - u8g2.getStrWidth(STATION_ID), 8, STATION_ID);
  u8g2.drawHLine(0, 10, 128);

  // ---- Rows 2-4: sensor values -----------------------------------------
  // font_6x10, 6px/char, max 21 chars per row -- every line fits:
  //   "T:28.5C"      = 7ch    "H:65%"   = 5ch  (right-aligned)
  //   "P:1013.3 hPa" = 13ch
  //   "Rain: DRY"    = 9ch
  u8g2.setFont(u8g2_font_6x10_tf);

  char tmpBuf[16], humBuf[12];
  sprintf(tmpBuf, "T:%.1fC",  disp.t_c);
  sprintf(humBuf, "H:%.0f%%", disp.rh_pct);
  u8g2.drawStr(0, 22, tmpBuf);
  u8g2.drawStr(127 - u8g2.getStrWidth(humBuf), 22, humBuf);  // right-align

  char presBuf[20];
  sprintf(presBuf, "P:%.1f hPa", disp.p_hpa);
  u8g2.drawStr(0, 35, presBuf);

  char rainBuf[20];
  if (disp.rain_detected) sprintf(rainBuf, "Rain:%d%%", disp.rain_pct);
  else                    strcpy(rainBuf, "Rain: DRY");
  u8g2.drawStr(0, 48, rainBuf);
  u8g2.drawHLine(0, 51, 128);

  // ---- Row 5: status (scroll if wider than 128px) ----------------------
  char statusBuf[48];
  if (disp.flagged) {
    if (strlen(disp.anomaly_var) > 0)
      sprintf(statusBuf, "!! ANOMALOUS [%s] !!", disp.anomaly_var);
    else
      strcpy(statusBuf, "!! ANOMALOUS !!");
  } else {
    strcpy(statusBuf, ">> NORMAL <<");
  }

  int sw = u8g2.getStrWidth(statusBuf);
  if (sw <= 128) {
    // Fits: center it; reset scroll for when anomaly appears later
    u8g2.drawStr((128 - sw) / 2, 62, statusBuf);
    scrollX = 128;
  } else {
    // Too wide: clip to bottom strip and scroll left
    u8g2.setClipWindow(0, 52, 127, 63);
    u8g2.drawStr(scrollX, 62, statusBuf);
    u8g2.setMaxClipWindow();
    if (now - lastScrollMs >= 50) {           // 1px / 50ms = 20px/s
      if (--scrollX < -(sw + 8)) scrollX = 128;
      lastScrollMs = now;
    }
  }

  u8g2.sendBuffer();
}

// ========================================================================
// SETUP
// ========================================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println(F("\n================================================"));
  Serial.print(F("  SkyGuard Edge - ")); Serial.println(STATION_ID);
  Serial.println(F("================================================"));

  dht.begin();
  Serial.println(F("DHT11 initialized."));

  pinMode(RAIN_DIGITAL_PIN, INPUT_PULLUP);
  pinMode(RAIN_ANALOG_PIN,  INPUT);
  analogReadResolution(12);
  Serial.println(F("Rain sensor initialized."));

  Wire.begin(21, 22);
  bmp_ready = bmp.begin();
  if (!bmp_ready) {
    Serial.println(F("\nERROR: BMP180 NOT FOUND!"));
    Serial.println(F("Check: SDA->GPIO21  SCL->GPIO22  VCC->3.3V  GND->GND"));
  } else {
    Serial.println(F("BMP180 initialized successfully."));
  }

  u8g2.begin();
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_7x13_tf);
  u8g2.drawStr(6, 35, "SkyGuard booting...");
  u8g2.sendBuffer();
  Serial.println(F("OLED initialized."));

  Serial.print(F("\nConnecting to WiFi: ")); Serial.println(WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500); Serial.print('.'); attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println(F("\nConnected to WiFi!"));
    Serial.print(F("Station at: http://"));
    Serial.print(WiFi.localIP());
    Serial.println(F("/data"));
  } else {
    Serial.println(F("\nWiFi FAILED. Check SSID/password."));
  }

  server.on("/",      HTTP_GET, handleData);   // root also returns JSON
  server.on("/data",  HTTP_GET, handleData);
  server.onNotFound(handleNotFound);
  server.begin();
  Serial.println(F("Web server started."));

  readSensorsAndEvaluate();   // seed payload immediately

  Serial.println(F("\nAll systems ready. Entering main loop.\n"));
}

// ========================================================================
// SENSOR READ + PHYSICS QC
// ========================================================================
void readSensorsAndEvaluate() {
  float tc = dht.readTemperature();
  float rh = dht.readHumidity();

  if (isnan(tc)) {
    if (bmp_ready) {
      tc = bmp.readTemperature();
      Serial.print(F("DHT11 temp fail -> BMP180: "));
      Serial.print(tc); Serial.println(F("C"));
    } else {
      tc = 28.5f;
      Serial.println(F("DHT11+BMP fail -> default 28.5C"));
    }
  }
  if (isnan(rh)) {
    rh = 45.0f;
    Serial.println(F("DHT11 hum fail -> default 45%"));
  }

  float p = 1013.25f;
  if (bmp_ready) p = bmp.readPressure() / 100.0f;

  bool rain_dig = (digitalRead(RAIN_DIGITAL_PIN) == LOW);
  int  rain_adc = analogRead(RAIN_ANALOG_PIN);
  int  rpct     = constrain(map(rain_adc, RAIN_DRY_RAW, RAIN_WET_RAW, 0, 100), 0, 100);
  bool rain_det = (rpct >= RAIN_DETECT_THRESHOLD_PCT);

  EquilibriumResult res;
  char anomaly_var[12] = "";

  if (!last_good.valid) {
    res.r_tv = res.r_n = res.r_e = res.composite = 0.0f;
    res.dalton_violation = res.flagged = false;
    last_good = { tc, p, rh, true };
    Serial.println(F("Baseline captured. QC starts next cycle."));
  } else {
    res = evaluateEquilibrium(tc, p, rh, last_good);
    if (res.flagged)
      classifyAnomalySource(tc, rh, p, last_good, anomaly_var);
    else
      last_good = { tc, p, rh, true };
  }

  disp.t_c             = tc;
  disp.rh_pct          = rh;
  disp.p_hpa           = p;
  disp.rain_detected   = rain_det;
  disp.rain_pct        = rpct;
  disp.flagged         = res.flagged;
  disp.dalton_violation= res.dalton_violation;
  disp.composite       = res.composite;
  strncpy(disp.anomaly_var, anomaly_var, sizeof(disp.anomaly_var) - 1);
  disp.anomaly_var[sizeof(disp.anomaly_var) - 1] = '\0';
  first_reading_done   = true;

  buildPayload(tc, p, rh, rain_det, rain_dig, rain_adc, rpct, res);

  Serial.print(STATION_ID);
  Serial.print(F("  T=")); Serial.print(tc, 1);
  Serial.print(F("C  RH=")); Serial.print(rh, 1);
  Serial.print(F("%  P=")); Serial.print(p, 1);
  Serial.print(F("hPa  Rain=")); Serial.print(rain_det ? "YES" : "no");
  Serial.print(F("(")); Serial.print(rpct);
  Serial.print(F("% adc=")); Serial.print(rain_adc);
  Serial.print(F(" dig=")); Serial.print(rain_dig ? "LOW" : "HIGH");
  Serial.print(F(")  comp=")); Serial.print(res.composite, 3);
  if (res.flagged) {
    Serial.print(F(" => ANOMALOUS [")); Serial.print(anomaly_var); Serial.println(F("]"));
  } else {
    Serial.println(F(" => normal"));
  }
}

// ========================================================================
// MAIN LOOP
// ========================================================================
void loop() {
  server.handleClient();

  unsigned long now = millis();
  if (now - lastSensorReadTime >= SENSOR_READ_INTERVAL) {
    lastSensorReadTime = now;
    readSensorsAndEvaluate();
  }

  drawOLED();   // internally rate-limited to ~30 FPS
}
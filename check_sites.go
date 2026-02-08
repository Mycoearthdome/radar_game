package main

import (
	"encoding/json"
	"fmt"
	"html/template"
	"math"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
)

// --- TYPES ---
type Radar struct {
	ID            string  `json:"id"`
	Type          string  `json:"type"`
	Lat           float64 `json:"lat"`
	Lon           float64 `json:"lon"`
	ElevationTier float64 `json:"elevation_tier"`
	TerrainHeight float64 `json:"terrain_height"`
	IsLive        bool    `json:"is_live"`
}

type City struct {
	Name string
	Lat  float64
	Lon  float64
}

// --- GLOBAL STATE ---
var (
	allRadars    []Radar
	liveRadars   []Radar
	cities       []City
	currentIndex = 0
	mu           sync.Mutex
	EarthRadius  = 6371.0
)

// Helper: Haversine distance
func getDistanceKM(lat1, lon1, lat2, lon2 float64) float64 {
	dLat := (lat2 - lat1) * math.Pi / 180.0
	dLon := (lon2 - lon1) * math.Pi / 180.0
	a := math.Sin(dLat/2)*math.Sin(dLat/2) +
		math.Cos(lat1*math.Pi/180.0)*math.Cos(lat2*math.Pi/180.0)*
			math.Sin(dLon/2)*math.Sin(dLon/2)
	return EarthRadius * 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
}

func main() {
	// 1. Load Radars
	data, err := os.ReadFile("RADAR.json")
	if err != nil {
		fmt.Println("Error: RADAR.json not found.")
		return
	}
	json.Unmarshal(data, &allRadars)

	// 2. Define Cities
	cityData := map[string][]float64{
		"Toronto": {43.65, -79.38}, "Montreal": {45.50, -73.56}, "Vancouver": {49.28, -123.12},
		"Calgary": {51.04, -114.07}, "Edmonton": {53.54, -113.49}, "Ottawa": {45.42, -75.69},
		"Winnipeg": {49.89, -97.13}, "Quebec City": {46.81, -71.20}, "Halifax": {44.64, -63.57},
		"Victoria": {48.42, -123.36}, "Saskatoon": {52.13, -106.67}, "St. John's": {47.56, -52.71},
		"Yellowknife": {62.45, -114.37}, "Whitehorse": {60.72, -135.05},
		"New York": {40.71, -74.00}, "Los Angeles": {34.05, -118.24}, "Chicago": {41.87, -87.62},
		"Houston": {29.76, -95.36}, "Washington DC": {38.89, -77.03}, "Miami": {25.76, -80.19},
		"Mexico City": {19.43, -99.13}, "Havana": {23.11, -82.36}, "Anchorage": {61.21, -149.90},
		"Sao Paulo": {-23.55, -46.63}, "Buenos Aires": {-34.60, -58.38}, "Lima": {-12.04, -77.04},
		"Bogota": {4.71, -74.07}, "Rio de Janeiro": {-22.90, -43.17}, "Santiago": {-33.44, -70.66},
		"Caracas": {10.48, -66.90}, "London": {51.50, -0.12}, "Paris": {48.85, 2.35},
		"Berlin": {52.52, 13.40}, "Moscow": {55.75, 37.61}, "Rome": {41.90, 12.49},
		"Madrid": {40.41, -3.70}, "Istanbul": {41.00, 28.97}, "Kyiv": {50.45, 30.52},
		"Stockholm": {59.32, 18.06}, "Warsaw": {52.22, 21.01}, "Beijing": {39.90, 116.40},
		"Tokyo": {35.68, 139.65}, "Seoul": {37.56, 126.97}, "Shanghai": {31.23, 121.47},
		"Hong Kong": {22.31, 114.16}, "Singapore": {1.35, 103.81}, "Mumbai": {19.07, 72.87},
		"New Delhi": {28.61, 77.20}, "Bangkok": {13.75, 100.50}, "Jakarta": {-6.20, 106.81},
		"Manila": {14.59, 120.98}, "Taipei": {25.03, 121.56}, "Astana": {51.16, 71.47},
		"Dubai": {25.20, 55.27}, "Riyadh": {24.71, 46.67}, "Tehran": {35.68, 51.38},
		"Tel Aviv": {32.08, 34.78}, "Cairo": {30.04, 31.23}, "Lagos": {6.52, 3.37},
		"Johannesburg": {-26.20, 28.04}, "Nairobi": {-1.29, 36.82}, "Casablanca": {33.57, -7.58},
		"Sydney": {-33.86, 151.20}, "Melbourne": {-37.81, 144.96}, "Auckland": {-36.84, 174.76},
		"Perth": {-31.95, 115.86}, "Honolulu": {21.30, -157.85},
	}
	for name, pos := range cityData {
		cities = append(cities, City{Name: name, Lat: pos[0], Lon: pos[1]})
	}

	// 3. API Endpoints
	http.HandleFunc("/get_next", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if currentIndex >= len(allRadars) {
			json.NewEncoder(w).Encode(map[string]interface{}{"status": "complete"})
			return
		}

		radar := allRadars[currentIndex]

		// Find nearest city
		nearestName := "Deep Wilderness"
		minDist := 999999.0
		for _, c := range cities {
			d := getDistanceKM(radar.Lat, radar.Lon, c.Lat, c.Lon)
			if d < minDist {
				minDist = d
				nearestName = c.Name
			}
		}

		json.NewEncoder(w).Encode(map[string]interface{}{
			"radar":        radar,
			"index":        currentIndex + 1,
			"total":        len(allRadars),
			"nearest_city": nearestName,
			"city_dist":    math.Round(minDist),
		})
	})

	http.HandleFunc("/mark_live", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if currentIndex < len(allRadars) {
			radar := allRadars[currentIndex]
			liveRadars = append(liveRadars, radar)
			currentIndex++
		}
		w.WriteHeader(http.StatusOK)
	})

	http.HandleFunc("/skip", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if currentIndex < len(allRadars) {
			currentIndex++
		}
		w.WriteHeader(http.StatusOK)
	})

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		template.Must(template.New("v").Parse(uiHTML)).Execute(w, nil)
	})

	go http.ListenAndServe(":8080", nil)
	fmt.Println("AEGIS Curation: http://localhost:8080")

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	<-sig

	mu.Lock()
	output, _ := json.MarshalIndent(liveRadars, "", "  ")
	os.WriteFile("checked_radars.json", output, 0644)
	fmt.Printf("\nSession Complete. %d radars exported.\n", len(liveRadars))
	mu.Unlock()
}

const uiHTML = `
<!DOCTYPE html>
<html>
<head>
    <title>AEGIS Strategic Console</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body { margin: 0; background: #000; color: #0f0; font-family: 'Courier New', monospace; overflow: hidden; }
        #map { height: 100vh; width: 100vw; z-index: 1; }
        #overlay { position: fixed; top: 0; width: 100%; height: 100%; z-index: 2000; pointer-events: none; }
        .hud { position: absolute; top: 20px; left: 50%; transform: translateX(-50%); background: rgba(0,15,0,0.95); 
               padding: 20px; border: 1px solid #0f0; pointer-events: auto; text-align: center; min-width: 500px; 
               box-shadow: 0 0 30px rgba(0,255,0,0.2); }
        #mini-map { position: fixed; bottom: 20px; right: 20px; width: 280px; height: 180px; 
                    border: 1px solid #0f0; z-index: 2500; background: #000; }
        .crosshair { position: fixed; top: 50%; left: 50%; width: 180px; height: 180px; 
                     border: 1px solid rgba(0,255,0,0.15); transform: translate(-50%, -50%); 
                     z-index: 1500; pointer-events: none; border-radius: 50%; }
        .crosshair::before, .crosshair::after { content: ""; position: absolute; background: rgba(255,0,0,0.3); }
        .crosshair::before { top: 50%; left: -20%; width: 140%; height: 1px; }
        .crosshair::after { left: 50%; top: -20%; height: 140%; width: 1px; }
        
        .objective { margin: 10px 0; padding: 10px; border-top: 1px solid #040; color: #0ff; font-weight: bold; }
        .controls { margin-top: 20px; display: flex; justify-content: center; gap: 30px; }
        button { font-family: monospace; padding: 12px 35px; cursor: pointer; font-weight: bold; border-radius: 2px; }
        .btn-live { background: #040; color: #0f0; border: 1px solid #0f0; }
        .btn-skip { background: #300; color: #f66; border: 1px solid #f66; }
        #progress-fill { height: 6px; background: #0f0; width: 0%; transition: width 0.4s; margin-top: 10px; }
    </style>
</head>
<body>
    <div id="overlay">
        <div class="hud">
            <div style="letter-spacing: 4px; font-size: 0.7em;">STRATEGIC DEFENSE CURATION</div>
            <h2 id="radar-id">RECOVERING DATA...</h2>
            <div id="radar-stats">---</div>
            <div class="objective">DEFENSIVE OBJECTIVE: <span id="target-city">---</span> (<span id="target-dist">---</span> KM)</div>
            <div id="progress-fill"></div>
            <div class="controls">
                <button class="btn-skip" onclick="skip()">[X] REJECT</button>
                <button class="btn-live" onclick="markLive()">[L] AUTHORIZE</button>
            </div>
        </div>
    </div>
    <div id="mini-map"></div>
    <div class="crosshair"></div>
    <div id="map"></div>

    <script>
        var map = L.map('map', {zoomControl: false, attributionControl: false}).setView([0, 0], 2);
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}').addTo(map);
        
        var miniMap = L.map('mini-map', {zoomControl: false, attributionControl: false, dragging: false, scrollWheelZoom: false}).setView([0, 0], 1);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png').addTo(miniMap);
        var miniMarker = L.circleMarker([0, 0], {color: '#0f0', radius: 4}).addTo(miniMap);

        async function loadNext() {
            const r = await fetch('/get_next');
            const data = await r.json();
            if (data.status === 'complete') {
                document.querySelector('.hud').innerHTML = "<h2>STACK VALIDATED</h2><p>CTRL+C in terminal to export checked_radars.json</p>";
                return;
            }

            const radar = data.radar;
            document.getElementById('radar-id').innerText = radar.id;
            document.getElementById('radar-stats').innerHTML = "COORD: " + radar.lat.toFixed(4) + ", " + radar.lon.toFixed(4) + " | TERRAIN: " + Math.round(radar.terrain_height) + "m";
            document.getElementById('target-city').innerText = data.nearest_city.toUpperCase();
            document.getElementById('target-dist').innerText = data.city_dist;
            document.getElementById('progress-fill').style.width = (data.index / data.total * 100) + "%";

            map.flyTo([radar.lat, radar.lon], 14, { duration: 1.5 });
            miniMap.setView([radar.lat, radar.lon], 1);
            miniMarker.setLatLng([radar.lat, radar.lon]);
        }

        async function markLive() { await fetch('/mark_live'); loadNext(); }
        async function skip() { await fetch('/skip'); loadNext(); }

        window.addEventListener('keydown', (e) => {
            if (e.key.toLowerCase() === 'l') markLive();
            if (e.key.toLowerCase() === 'x') skip();
        });

        setTimeout(loadNext, 500);
    </script>
</body>
</html>`

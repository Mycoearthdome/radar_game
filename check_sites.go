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

var (
	allRadars    []Radar
	liveRadars   []Radar
	cities       []City
	currentIndex = 0
	mu           sync.Mutex
	EarthRadius  = 6371.0
)

func getDistanceKM(lat1, lon1, lat2, lon2 float64) float64 {
	dLat := (lat2 - lat1) * math.Pi / 180.0
	dLon := (lon2 - lon1) * math.Pi / 180.0
	a := math.Sin(dLat/2)*math.Sin(dLat/2) +
		math.Cos(lat1*math.Pi/180.0)*math.Cos(lat2*math.Pi/180.0)*
			math.Sin(dLon/2)*math.Sin(dLon/2)
	return EarthRadius * 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
}

func main() {
	raw, err := os.ReadFile("RADAR.json")
	if err != nil {
		fmt.Println("Error: RADAR.json not found.")
		return
	}

	var rawData []map[string]interface{}
	if err := json.Unmarshal(raw, &rawData); err != nil {
		fmt.Printf("JSON Error: %v\n", err)
		return
	}

	for _, item := range rawData {
		// Helper to safely extract float64 or default to 0
		getFloat := func(key string) float64 {
			if val, ok := item[key].(float64); ok {
				return val
			}
			return 0.0
		}

		typ, _ := item["type"].(string)
		id, _ := item["id"].(string)
		lat := getFloat("lat")
		lon := getFloat("lon")

		if typ == "CITY" {
			cities = append(cities, City{Name: id, Lat: lat, Lon: lon})
		} else if typ == "RADAR" {
			allRadars = append(allRadars, Radar{
				ID:            id,
				Type:          typ,
				Lat:           lat,
				Lon:           lon,
				ElevationTier: getFloat("elevation_tier"), // Safe conversion
				TerrainHeight: getFloat("terrain_height"), // Safe conversion
				IsLive:        false,
			})
		}
	}

	http.HandleFunc("/get_next", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if currentIndex >= len(allRadars) {
			json.NewEncoder(w).Encode(map[string]interface{}{"status": "complete"})
			return
		}
		radar := allRadars[currentIndex]
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
			radar.IsLive = true
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
	fmt.Printf("AEGIS Curation Initialized: %d Radars, %d Cities\n", len(allRadars), len(cities))

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	<-sig

	mu.Lock()
	output, _ := json.MarshalIndent(liveRadars, "", "  ")
	os.WriteFile("checked_radars.json", output, 0644)
	fmt.Printf("\nExported %d radars.\n", len(liveRadars))
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

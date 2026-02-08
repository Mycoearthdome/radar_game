package main

import (
	"encoding/json"
	"fmt"
	"html/template"
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

var (
	allRadars    []Radar
	liveRadars   []Radar
	currentIndex = 0
	mu           sync.Mutex
)

func main() {
	// 1. Load the optimized radars
	data, err := os.ReadFile("RADAR.json")
	if err != nil {
		fmt.Println("Error: RADAR.json not found. Run the simulation first.")
		return
	}
	json.Unmarshal(data, &allRadars)
	fmt.Printf("Curation Mode: %d radars to review. http://localhost:8080\n", len(allRadars))

	// 2. API Endpoints
	http.HandleFunc("/get_next", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if currentIndex >= len(allRadars) {
			json.NewEncoder(w).Encode(map[string]interface{}{"status": "complete"})
			return
		}
		json.NewEncoder(w).Encode(map[string]interface{}{
			"radar": allRadars[currentIndex],
			"index": currentIndex + 1,
			"total": len(allRadars),
		})
	})

	http.HandleFunc("/mark_live", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if currentIndex < len(allRadars) {
			radar := allRadars[currentIndex]
			radar.IsLive = true
			liveRadars = append(liveRadars, radar)
			fmt.Printf("✓ [%d/%d] Radar %s marked LIVE\n", currentIndex+1, len(allRadars), radar.ID)
			currentIndex++
		}
		w.WriteHeader(http.StatusOK)
	})

	http.HandleFunc("/skip", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if currentIndex < len(allRadars) {
			fmt.Printf("⨯ [%d/%d] Radar %s SKIPPED\n", currentIndex+1, len(allRadars), allRadars[currentIndex].ID)
			currentIndex++
		}
		w.WriteHeader(http.StatusOK)
	})

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		template.Must(template.New("v").Parse(uiHTML)).Execute(w, nil)
	})

	go http.ListenAndServe(":8080", nil)

	// 3. CTRL+C to Save
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	<-sig

	mu.Lock()
	output, _ := json.MarshalIndent(liveRadars, "", "  ")
	os.WriteFile("checked_radars.json", output, 0644)
	fmt.Printf("\nSaved %d validated radars to checked_radars.json. Session ended.\n", len(liveRadars))
	mu.Unlock()
}

const uiHTML = `
<!DOCTYPE html>
<html>
<head>
    <title>AEGIS - Curation Console</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body { margin: 0; background: #000; color: #0f0; font-family: 'Courier New', monospace; overflow: hidden; }
        #overlay { position: fixed; top: 0; width: 100%; height: 100%; z-index: 2000; pointer-events: none; }
        .hud { position: absolute; top: 20px; left: 50%; transform: translateX(-50%); background: rgba(0,20,0,0.85); 
               padding: 20px; border: 1px solid #0f0; pointer-events: auto; text-align: center; min-width: 400px; }
        #map { height: 100vh; width: 100vw; }
        .crosshair { position: fixed; top: 50%; left: 50%; width: 120px; height: 120px; 
                     border: 1px solid rgba(255,255,255,0.3); transform: translate(-50%, -50%); 
                     z-index: 1500; pointer-events: none; border-radius: 50%; }
        .crosshair::before { content: ""; position: absolute; top: 50%; left: -10%; width: 120%; height: 1px; background: rgba(255,0,0,0.5); }
        .crosshair::after { content: ""; position: absolute; left: 50%; top: -10%; height: 120%; width: 1px; background: rgba(255,0,0,0.5); }
        
        .controls { margin-top: 15px; display: flex; justify-content: center; gap: 20px; }
        button { font-family: monospace; padding: 10px 30px; cursor: pointer; font-weight: bold; font-size: 1.1em; transition: 0.2s; }
        .btn-live { background: #040; color: #0f0; border: 1px solid #0f0; }
        .btn-live:hover { background: #0f0; color: #000; box-shadow: 0 0 15px #0f0; }
        .btn-skip { background: #400; color: #f66; border: 1px solid #f66; }
        .btn-skip:hover { background: #f66; color: #000; box-shadow: 0 0 15px #f66; }
        
        #progress-bar { width: 100%; height: 4px; background: #111; margin-top: 10px; }
        #progress-fill { height: 100%; background: #0f0; width: 0%; transition: width 0.3s; }
    </style>
</head>
<body>
    <div id="overlay">
        <div class="hud">
            <div style="font-size: 0.8em; opacity: 0.7;">AEGIS CURATION INTERFACE</div>
            <h2 id="radar-id">SCANNING...</h2>
            <div id="radar-stats">---</div>
            <div id="progress-bar"><div id="progress-fill"></div></div>
            <div style="margin-top: 5px; font-size: 0.7em;" id="counter">RADAR 0 OF 0</div>
            
            <div class="controls">
                <button class="btn-skip" onclick="event.stopPropagation(); skip();">SKIP [X]</button>
                <button class="btn-live" onclick="event.stopPropagation(); markLive();">ATTRIBUTE LIVE [L]</button>
            </div>
        </div>
    </div>
    <div class="crosshair"></div>
    <div id="map"></div>

    <script>
        var map = L.map('map', {zoomControl: false, attributionControl: false}).setView([0, 0], 2);
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}').addTo(map);
        
        let active = false;

        async function loadNext() {
            const r = await fetch('/get_next');
            const data = await r.json();
            
            if (data.status === 'complete') {
                document.querySelector('.hud').innerHTML = "<h2>VERIFICATION COMPLETE</h2><p>Press CTRL+C in your Terminal to save checked_radars.json</p>";
                return;
            }

            const radar = data.radar;
            document.getElementById('radar-id').innerText = radar.id;
            document.getElementById('radar-stats').innerText = 
                "LAT: " + radar.lat.toFixed(4) + " LON: " + radar.lon.toFixed(4) + " | TERRAIN: " + Math.round(radar.terrain_height) + "m";
            
            document.getElementById('counter').innerText = "RADAR " + data.index + " OF " + data.total;
            document.getElementById('progress-fill').style.width = (data.index / data.total * 100) + "%";

            map.flyTo([radar.lat, radar.lon], 14, { duration: 1.5 });
            active = true;
        }

        async function markLive() {
            if (!active) return; active = false;
            await fetch('/mark_live');
            loadNext();
        }

        async function skip() {
            if (!active) return; active = false;
            await fetch('/skip');
            loadNext();
        }

        // Keyboard Shortcuts
        window.addEventListener('keydown', (e) => {
            if (e.key.toLowerCase() === 'l') markLive();
            if (e.key.toLowerCase() === 'x') skip();
        });

        // Click map to attribute live
        map.on('click', markLive);

        setTimeout(loadNext, 500);
    </script>
</body>
</html>`

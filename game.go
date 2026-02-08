package main

import (
	"encoding/json"
	"fmt"
	"html/template"
	"math"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

// --- CONFIGURATION ---
const (
	RadarRadiusKM   = 1200.0
	SatRadiusKM     = 1400.0
	RadarFile       = "RADAR.json"
	SatCost         = 400.0
	InterceptGain   = 25.0
	BasePeaceDiv    = 25.0
	MaxSatellites   = 45
	EmergencyTarget = 2 * time.Minute
	EarthRadius     = 6371.0
)

var (
	BaseWarp        = 50.0
	CurrentWarp     = 50.0
	entities        = make(map[string]*Entity)
	budget          = 3000.0
	mu              sync.RWMutex
	lastImpact      time.Time
	simClock        time.Time
	currentCycle    = 1
	activeTarget    = EmergencyTarget
	radarEfficiency = make(map[string]int)
	cityNames       []string
)

type Entity struct {
	ID       string  `json:"id"`
	Type     string  `json:"type"`
	Lat      float64 `json:"lat"`
	Lon      float64 `json:"lon"`
	TargetID string  `json:"target_id,omitempty"`
	Phase    float64 `json:"phase"`
}

type Command struct {
	Action string
	ID     string
	Data   *Entity
}

func getDistanceKM(lat1, lon1, lat2, lon2 float64) float64 {
	dLat := (lat2 - lat1) * math.Pi / 180.0
	dLon := (lon2 - lon1) * math.Pi / 180.0
	a := math.Sin(dLat/2)*math.Sin(dLat/2) +
		math.Cos(lat1*math.Pi/180.0)*math.Cos(lat2*math.Pi/180.0)*
			math.Sin(dLon/2)*math.Sin(dLon/2)
	return EarthRadius * 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
}

func setupSimulation() {
	cities := map[string][]float64{
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
	for name, pos := range cities {
		entities[name] = &Entity{ID: name, Type: "CITY", Lat: pos[0], Lon: pos[1]}
		cityNames = append(cityNames, name)
	}
	if data, err := os.ReadFile(RadarFile); err == nil {
		var saved []Entity
		if err := json.Unmarshal(data, &saved); err == nil {
			for i := range saved {
				entities[saved[i].ID] = &saved[i]
				radarEfficiency[saved[i].ID] = 1
			}
		}
	}
	simClock = time.Now()
	lastImpact = simClock.Add(-5 * time.Second)
}

func autonomousEraReset() {
	mu.Lock()
	defer mu.Unlock()

	fmt.Printf("\n--- ERA %d ANALYSIS ---\n", currentCycle)
	survivors := 0
	totalRadars := 0

	for id, e := range entities {
		if e.Type == "RADAR" {
			totalRadars++
			if radarEfficiency[id] == 0 {
				delete(entities, id)
			} else {
				radarEfficiency[id] = 0 // Persistent: score reset for new era test
				survivors++
			}
		}
		if e.Type == "MISSILE" || e.Type == "INTERCEPTOR" {
			delete(entities, id)
		}
	}

	currentCycle++
	budget += 5000.0
	lastImpact = simClock
	fmt.Printf("Evolution: %d/%d Radars retained. New Era %d Starting.\n", survivors, totalRadars, currentCycle)
}

func saveOptimalAndExit() {
	mu.Lock()
	defer mu.Unlock()
	var optimal []Entity
	for _, e := range entities {
		if e.Type == "RADAR" {
			optimal = append(optimal, *e)
		}
	}
	data, _ := json.MarshalIndent(optimal, "", "  ")
	os.WriteFile(RadarFile, data, 0644)
	fmt.Printf("\nShutting down. Optimized Grid Saved: %d Radars.\n", len(optimal))
}

func runPhysicsEngine() {
	for {
		commands := []Command{}
		mu.Lock()

		if simClock.Sub(lastImpact) >= (time.Duration(currentCycle) * activeTarget) {
			mu.Unlock()
			autonomousEraReset()
			continue
		}

		threats, satCount := 0, 0
		for _, e := range entities {
			if e.Type == "MISSILE" {
				threats++
			}
			if e.Type == "SATELLITE" {
				satCount++
			}
		}

		if threats < (currentCycle+3) && (rand.Float64() < 0.1 || threats == 0) {
			mID := fmt.Sprintf("M-%d-%d", simClock.Unix(), rand.Intn(1000))
			target := cityNames[rand.Intn(len(cityNames))]
			entities[mID] = &Entity{ID: mID, Type: "MISSILE", Lat: rand.Float64()*150 - 75, Lon: rand.Float64()*360 - 180, TargetID: target}
		}

		CurrentWarp = BaseWarp
		simClock = simClock.Add(time.Duration(CurrentWarp) * time.Second)
		budget += (BasePeaceDiv * CurrentWarp / 3600.0)

		for id, e := range entities {
			switch e.Type {
			case "SATELLITE":
				e.Lon += 0.8 * (CurrentWarp / 10.0)
				e.Lat = 72.0 * math.Sin((math.Pi/180.0)*e.Lon+e.Phase)
				if e.Lon > 180 {
					e.Lon -= 360
				}
			case "MISSILE":
				target := entities[e.TargetID]
				if getDistanceKM(e.Lat, e.Lon, target.Lat, target.Lon) < (35.0 * CurrentWarp) {
					lastImpact = simClock
					budget -= 1200.0

					// Saturation Logic
					isSaturated := false
					for _, existing := range entities {
						if existing.Type == "RADAR" && getDistanceKM(e.Lat, e.Lon, existing.Lat, existing.Lon) < (RadarRadiusKM*0.75) {
							isSaturated = true
							break
						}
					}

					if !isSaturated {
						rID := fmt.Sprintf("RADAR-%d", rand.Intn(1e9))
						commands = append(commands, Command{"ADD", rID, &Entity{ID: rID, Type: "RADAR", Lat: e.Lat, Lon: e.Lon}})
						radarEfficiency[rID] = 0
					}
					commands = append(commands, Command{"DEL", id, nil})
				} else {
					e.Lat += (target.Lat - e.Lat) * 0.04
					e.Lon += (target.Lon - e.Lon) * 0.04
					for sid, s := range entities {
						if (s.Type == "RADAR" || s.Type == "SATELLITE") && getDistanceKM(e.Lat, e.Lon, s.Lat, s.Lon) < RadarRadiusKM {
							if s.Type == "RADAR" {
								radarEfficiency[sid]++
							}
							iID := "I-" + id
							if _, t := entities[iID]; !t {
								commands = append(commands, Command{"ADD", iID, &Entity{ID: iID, Type: "INTERCEPTOR", Lat: s.Lat, Lon: s.Lon, TargetID: id}})
							}
							break
						}
					}
				}
			case "INTERCEPTOR":
				target, ok := entities[e.TargetID]
				if !ok {
					commands = append(commands, Command{"DEL", id, nil})
					continue
				}
				if getDistanceKM(e.Lat, e.Lon, target.Lat, target.Lon) < (45.0 * CurrentWarp) {
					budget += InterceptGain
					commands = append(commands, Command{"DEL", e.TargetID, nil})
					commands = append(commands, Command{"DEL", id, nil})
				} else {
					e.Lat += (target.Lat - e.Lat) * 0.35
					e.Lon += (target.Lon - e.Lon) * 0.35
				}
			}
		}

		for _, cmd := range commands {
			if cmd.Action == "ADD" {
				entities[cmd.ID] = cmd.Data
			}
			if cmd.Action == "DEL" {
				delete(entities, cmd.ID)
			}
		}

		if satCount < MaxSatellites && budget >= SatCost {
			budget -= SatCost
			sID := fmt.Sprintf("S-%d", rand.Intn(1e6))
			site := launchSites[rand.Intn(len(launchSites))]
			entities[sID] = &Entity{ID: sID, Type: "SATELLITE", Lat: site.Lat, Lon: site.Lon, Phase: float64(satCount) * 8.0}
		}
		mu.Unlock()
		time.Sleep(10 * time.Millisecond)
	}
}

var launchSites = []LaunchSite{
	{"KSC", 28.57, -80.64}, {"Baikonur", 45.96, 63.30}, {"Guiana", 5.23, -52.76},
}

type LaunchSite struct {
	Name     string
	Lat, Lon float64
}

func main() {
	setupSimulation()
	go runPhysicsEngine()

	http.HandleFunc("/intel", func(w http.ResponseWriter, r *http.Request) {
		mu.RLock()
		defer mu.RUnlock()
		var all []Entity
		for _, e := range entities {
			all = append(all, *e)
		}
		json.NewEncoder(w).Encode(map[string]interface{}{
			"peace": simClock.Sub(lastImpact).Seconds(), "budget": budget,
			"cycle": currentCycle, "target": (time.Duration(currentCycle) * activeTarget).Seconds(),
			"entities": all,
		})
	})

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		template.Must(template.New("v").Parse(uiHTML)).Execute(w, nil)
	})

	go http.ListenAndServe(":8080", nil)
	fmt.Println("AEGIS V9.5 ACTIVE: http://localhost:8080")

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	<-sig
	saveOptimalAndExit()
}

const uiHTML = `
<!DOCTYPE html>
<html>
<head>
    <title>AEGIS V9.5</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body { margin: 0; background: #000; color: #0f0; font-family: monospace; overflow: hidden; }
        #stats { position: absolute; top: 0; width: 100%; display: flex; justify-content: space-around; background: rgba(0,10,0,0.9); padding: 12px; z-index: 1000; border-bottom: 1px solid #0f0; }
        #map { height: 100vh; width: 100vw; background: #000; }
        #progress-bar { height: 4px; width: 0%; background: #0ff; position: absolute; top: 50px; z-index: 1001; transition: width 0.1s linear; }
    </style>
</head>
<body>
    <div id="stats">
        <div>ERA: <span id="cycle-num">1</span></div>
        <div>PEACE_TICK: <span id="cur">0s</span></div>
        <div>FUNDS: <span id="funds">$0</span></div>
    </div>
    <div id="progress-bar"></div>
    <div id="map"></div>
    <script>
        var map = L.map('map', {zoomControl: false, attributionControl: false}).setView([20, 0], 2);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png').addTo(map);
        var layers = {};
        
        async function sync() {
            try {
                const r = await fetch('/intel'); const d = await r.json();
                document.getElementById('cur').innerText = Math.floor(d.peace) + "s";
                document.getElementById('funds').innerText = "$" + Math.floor(d.budget);
                document.getElementById('cycle-num').innerText = d.cycle;
                document.getElementById('progress-bar').style.width = (d.peace / d.target) * 100 + "%";

                d.entities.forEach(e => {
                    if (!layers[e.id]) {
                        if (e.type === 'SATELLITE') layers[e.id] = L.circleMarker([e.lat, e.lon], {color: '#0ff', radius: 3}).addTo(map);
                        if (e.type === 'RADAR') layers[e.id] = L.circle([e.lat, e.lon], {color: '#0f0', radius: 1200000, fillOpacity: 0.1, weight: 1}).addTo(map);
                        if (e.type === 'MISSILE') layers[e.id] = L.circleMarker([e.lat, e.lon], {color: '#f00', radius: 4}).addTo(map);
                        if (e.type === 'INTERCEPTOR') layers[e.id] = L.circleMarker([e.lat, e.lon], {color: '#fff', radius: 2}).addTo(map);
                        if (e.type === 'CITY') layers[e.id] = L.marker([e.lat, e.lon], {icon: L.divIcon({html: '<div style="background:#ff0;width:3px;height:3px;"></div>'})}).addTo(map);
                    } else if (layers[e.id].setLatLng) {
                        layers[e.id].setLatLng([e.lat, e.lon]);
                    }
                });

                const currentIds = new Set(d.entities.map(e => e.id));
                Object.keys(layers).forEach(id => {
                    if (!currentIds.has(id)) { map.removeLayer(layers[id]); delete layers[id]; }
                });
            } catch(e) {}
        }
        setInterval(sync, 100);
    </script>
</body>
</html>`

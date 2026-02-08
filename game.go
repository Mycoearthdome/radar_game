package main

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"html/template"
	"math"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"sync"
	"syscall"
	"time"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

const (
	RadarRadiusKM     = 1200.0
	RadarCost         = 500.0
	MaxRadars         = 500
	CityImpactPenalty = 5000.0
	InterceptReward   = 1500.0 // TRIPLED: Makes radars highly profitable to encourage building
	EarthRadius       = 6371.0
	EraDuration       = 24 * time.Hour
	RadarFile         = "RADAR.json"
	BrainFile         = "BRAIN.gob"
)

var (
	entities        = make(map[string]*Entity)
	kills           = make(map[string]int)
	threatHistory   = make([]Coordinate, 0)
	lastVectors     = make([]Vector, 0)
	budget          = 200000.0
	mu              sync.RWMutex
	eraStartTime    time.Time
	simClock        time.Time
	currentCycle    = 1
	brain           *Brain
	successRate     float64
	totalThreats    int
	totalIntercepts int
	minRadarEver    = MaxRadars
	forceReset      = false
	cityNames       []string
)

type Entity struct {
	ID    string  `json:"id" gob:"id"`
	Type  string  `json:"type" gob:"type"`
	Lat   float64 `json:"lat" gob:"lat"`
	Lon   float64 `json:"lon" gob:"lon"`
	BaseL float64 `json:"base_lat" gob:"base_lat"`
	Phase float64 `json:"phase" gob:"phase"`
}

type Coordinate struct {
	Lat float64 `json:"lat"`
	Lon float64 `json:"lon"`
}

type Vector struct {
	Start Coordinate `json:"start"`
	End   Coordinate `json:"end"`
}

// --- BRAIN ENGINE ---
type Brain struct {
	g      *gorgonia.ExprGraph
	w0, w1 *gorgonia.Node
}

func NewBrain() *Brain {
	g := gorgonia.NewGraph()
	w0 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(3, 12), gorgonia.WithName("w0"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	w1 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(12, 3), gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	return &Brain{g: g, w0: w0, w1: w1}
}

func (b *Brain) Predict(input []float64) int {
	g := gorgonia.NewGraph()
	x := gorgonia.NewVector(g, tensor.Float64, gorgonia.WithShape(3), gorgonia.WithValue(tensor.New(tensor.WithBacking(input))))
	w0 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(3, 12), gorgonia.WithValue(b.w0.Value()))
	w1 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(12, 3), gorgonia.WithValue(b.w1.Value()))
	l0 := gorgonia.Must(gorgonia.Rectify(gorgonia.Must(gorgonia.Mul(x, w0))))
	out := gorgonia.Must(gorgonia.SoftMax(gorgonia.Must(gorgonia.Mul(l0, w1))))
	vm := gorgonia.NewTapeMachine(g)
	defer vm.Close()
	vm.RunAll()
	res := out.Value().Data().([]float64)
	maxIdx := 0
	for i, v := range res {
		if v > res[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}

func (b *Brain) SaveWeights() {
	f, _ := os.Create(BrainFile)
	defer f.Close()
	gob.NewEncoder(f).Encode(b.w0.Value())
	gob.NewEncoder(f).Encode(b.w1.Value())
}

func (b *Brain) LoadWeights() {
	f, err := os.Open(BrainFile)
	if err != nil {
		return
	}
	defer f.Close()
	dec := gob.NewDecoder(f)
	var w0T, w1T *tensor.Dense
	dec.Decode(&w0T)
	dec.Decode(&w1T)
	if w0T != nil {
		gorgonia.Let(b.w0, w0T)
	}
	if w1T != nil {
		gorgonia.Let(b.w1, w1T)
	}
}

// --- SYSTEM LOGICS ---

func saveSystemState() {
	brain.SaveWeights()
	var opt []Entity
	for _, e := range entities {
		if e.Type == "CITY" || e.Type == "RADAR" {
			opt = append(opt, *e)
		}
	}
	d, _ := json.Marshal(opt)
	os.WriteFile(RadarFile, d, 0644)
}

func autonomousEraReset() {
	mu.Lock()
	defer mu.Unlock()

	currentCycle++
	eraStartTime = simClock

	rCount := 0
	for id, e := range entities {
		if e.Type == "RADAR" {
			rCount++
			if kills[id] < 1 {
				delete(entities, id)
				delete(kills, id)
				rCount--
			} else {
				kills[id] = 0
			}
		}
	}

	if successRate >= 95.0 && rCount <= minRadarEver {
		minRadarEver = rCount
		fmt.Printf("[ERA %d] RECORD: %d Radars @ %.1f%% Success. Saving...\n", currentCycle-1, rCount, successRate)
		saveSystemState()
	}

	budget = 200000.0
	totalThreats = 0
	totalIntercepts = 0
}

func getDistanceKM(lat1, lon1, lat2, lon2 float64) float64 {
	p1, p2 := lat1*math.Pi/180, lat2*math.Pi/180
	dp, dl := (lat2-lat1)*math.Pi/180, (lon2-lon1)*math.Pi/180
	a := math.Sin(dp/2)*math.Sin(dp/2) + math.Cos(p1)*math.Cos(p2)*math.Sin(dl/2)*math.Sin(dl/2)
	return EarthRadius * 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
}

func runPhysicsEngine() {
	for {
		mu.Lock()
		if simClock.After(eraStartTime.Add(EraDuration)) || forceReset {
			forceReset = false
			mu.Unlock()
			autonomousEraReset()
			continue
		}

		rCount := 0
		for _, e := range entities {
			if e.Type == "RADAR" {
				rCount++
			}
		}

		// Threats
		if rand.Float64() < 0.95 {
			totalThreats++
			target := entities[cityNames[rand.Intn(len(cityNames))]]
			origin := Coordinate{Lat: rand.Float64()*180 - 90, Lon: rand.Float64()*360 - 180}
			intercepted := false
			for id, e := range entities {
				if e.Type == "RADAR" && getDistanceKM(e.Lat, e.Lon, target.Lat, target.Lon) < RadarRadiusKM {
					kills[id]++
					budget += InterceptReward
					totalIntercepts++
					intercepted = true
					break
				}
			}

			if len(lastVectors) < 10 {
				lastVectors = append(lastVectors, Vector{Start: origin, End: Coordinate{Lat: target.Lat, Lon: target.Lon}})
			} else {
				lastVectors[totalThreats%10] = Vector{Start: origin, End: Coordinate{Lat: target.Lat, Lon: target.Lon}}
			}

			if !intercepted {
				budget -= CityImpactPenalty
				threatHistory = append(threatHistory, Coordinate{Lat: target.Lat, Lon: target.Lon})
				if len(threatHistory) > 30 {
					threatHistory = threatHistory[1:]
				}
			}
		}

		if totalThreats > 0 {
			successRate = (float64(totalIntercepts) / float64(totalThreats)) * 100
		}

		// STAGNATION BREAKER: Force exploration if NN is stuck at 1 radar
		input := []float64{
			math.Max(-1, math.Min(budget/200000.0, 1.0)),
			float64(rCount) / 10.0, // Scale sensitivity
			successRate / 100.0,
		}
		action := brain.Predict(input)

		// Overriding the NN if success is low OR budget is massive OR random 2% chance
		shouldBuild := action == 1 || (successRate < 80.0 && budget > 0) || (rand.Float64() < 0.02)

		if shouldBuild && budget >= RadarCost && rCount < MaxRadars {
			id := fmt.Sprintf("R-%d", rand.Intn(1e6))
			c := entities[cityNames[rand.Intn(len(cityNames))]]
			entities[id] = &Entity{
				ID: id, Type: "RADAR",
				Lat: c.Lat + (rand.Float64()-0.5)*10,
				Lon: c.Lon + (rand.Float64()-0.5)*10,
			}
			budget -= RadarCost
		}

		simClock = simClock.Add(1 * time.Hour)
		mu.Unlock()
		runtime.Gosched()
	}
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
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
		entities[name] = &Entity{ID: name, Type: "CITY", Lat: pos[0], Lon: pos[1]}
		cityNames = append(cityNames, name)
	}

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
			"cycle": currentCycle, "budget": budget, "entities": all, "success": successRate, "min_ever": minRadarEver, "vectors": lastVectors,
		})
	})
	http.HandleFunc("/skip", func(w http.ResponseWriter, r *http.Request) { mu.Lock(); forceReset = true; mu.Unlock() })
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		template.Must(template.New("v").Parse(uiHTML)).Execute(w, nil)
	})

	go http.ListenAndServe(":8080", nil)
	fmt.Println("AEGIS V11.4.3: Reinforced Incentives Online.")

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	<-stop
	mu.Lock()
	saveSystemState()
	mu.Unlock()
	fmt.Println("Safe Exit.")
}

func setupSimulation() {
	simClock, eraStartTime = time.Now(), time.Now()
	if data, err := os.ReadFile(RadarFile); err == nil {
		var saved []Entity
		json.Unmarshal(data, &saved)
		for _, e := range saved {
			if e.Type != "CITY" {
				entities[e.ID] = &e
				kills[e.ID] = 1
			}
		}
	}
	brain = NewBrain()
	brain.LoadWeights()
}

const uiHTML = `
<!DOCTYPE html><html><head><title>AEGIS V11.4.3</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
    body { margin:0; background:#000; color:#0f0; font-family:monospace; }
    #stats { position:fixed; top:10px; left:10px; z-index:1000; background:rgba(0,10,20,0.9); padding:10px; border:1px solid #0af; font-size:12px; line-height:1.5;}
    #map { height:100vh; width:100vw; }
    button { background:#0af; color:#000; border:none; padding:4px; cursor:pointer; font-weight:bold; width:100%; margin-top:5px;}
</style></head>
<body>
    <div id="stats">
        ERA: <span id="era">0</span><br>
        RADARS: <span id="rcount">0</span><br>
        BEST MIN: <span id="minr">0</span><br>
        SUCCESS: <span id="success">0</span>%<br>
        BUDGET: $<span id="budget">0</span>
        <button onclick="fetch('/skip')">MANUAL ERA SKIP</button>
    </div>
    <div id="map"></div>
<script>
    var map = L.map('map', {zoomControl:false, attributionControl:false}).setView([20, 0], 2);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png').addTo(map);
    var layers = {}, vectorLayers = [];
    async function sync() {
        try {
            const r = await fetch('/intel'); const d = await r.json();
            document.getElementById('era').innerText = d.cycle;
            document.getElementById('success').innerText = d.success.toFixed(1);
            document.getElementById('minr').innerText = d.min_ever;
            document.getElementById('budget').innerText = d.budget.toLocaleString();
            
            vectorLayers.forEach(l => map.removeLayer(l)); vectorLayers = [];
            d.vectors.forEach(v => {
                vectorLayers.push(L.polyline([[v.start.lat, v.start.lon], [v.end.lat, v.end.lon]], {color: 'red', weight: 1, opacity: 0.2}).addTo(map));
            });

            const currentIDs = d.entities.map(e => e.id);
            Object.keys(layers).forEach(id => { if(!currentIDs.includes(id)) { map.removeLayer(layers[id]); delete layers[id]; } });
            d.entities.forEach(e => {
                if (!layers[e.id]) {
                    if (e.type === 'RADAR') layers[e.id] = L.circle([e.lat, e.lon], {radius:1200000, color:'#0f0', weight:0.5, fillOpacity:0.05}).addTo(map);
                    else if (e.type === 'CITY') layers[e.id] = L.circleMarker([e.lat, e.lon], {radius:4, color:'#f00'}).addTo(map);
                } else { layers[e.id].setLatLng([e.lat, e.lon]); }
            });
            document.getElementById('rcount').innerText = d.entities.filter(e => e.type === 'RADAR').length;
        } catch(e) {}
    }
    setInterval(sync, 1000);
</script></body></html>`

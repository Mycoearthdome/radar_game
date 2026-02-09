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
	"runtime"
	"sort"
	"sync"
	"time"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

const (
	RadarRadiusKM     = 1200.0
	TetherRadiusKM    = 800.0
	RadarCost         = 500.0
	RelocationCost    = 50.0
	EnforcementFee    = 1000.0
	MinRadars         = 60
	CityImpactPenalty = 4500.0
	InterceptReward   = 6000.0
	EfficiencyBonus   = 35000.0
	EarthRadius       = 6371.0
	EraDuration       = 24 * time.Hour
	RadarFile         = "RADAR.json"
	BrainFile         = "BRAIN.gob"
	BankruptcyLimit   = 1500000.0
)

var (
	entities         = make(map[string]*Entity)
	kills            = make(map[string]int)
	budget           = 2000000.0
	totalEraSpending = 0.0
	mu               sync.RWMutex
	eraStartTime     time.Time
	simClock         time.Time
	wallStart        time.Time // For Benchmark
	currentCycle     = 1
	brain            *Brain
	successRate      float64
	totalThreats     int
	totalIntercepts  int
	MaxRadars        = 500
	minEverRadars    = MaxRadars
	forceReset       = false
	cityNames        []string
	mutationRate     = 1.0
	quadrantMisses   = make([]float64, 4) // [NW, NE, SW, SE]
)

type Entity struct {
	ID        string  `json:"id" gob:"id"`
	Type      string  `json:"type" gob:"type"`
	Lat       float64 `json:"lat" gob:"lat"`
	Lon       float64 `json:"lon" gob:"lon"`
	LastMoved int64   `json:"last_moved"`
}

type Brain struct {
	g      *gorgonia.ExprGraph
	w0, w1 *gorgonia.Node
	x      *gorgonia.Node
	out    *gorgonia.Node
	vm     gorgonia.VM
}

func (b *Brain) Mutate(rate float64) {
	// Access the underlying tensors for weights
	w0Data := b.w0.Value().Data().([]float64)
	w1Data := b.w1.Value().Data().([]float64)

	// Apply Gaussian noise: weight = weight + (randn * rate)
	for i := range w0Data {
		w0Data[i] += rand.NormFloat64() * rate
	}
	for i := range w1Data {
		w1Data[i] += rand.NormFloat64() * rate
	}
}

func NewBrain(multiplier float64) *Brain {
	g := gorgonia.NewGraph()
	// Input: [Budget/Limit, RadarCount/Max, SuccessRate, MissNW, MissNE, MissSW, MissSE]
	x := gorgonia.NewVector(g, tensor.Float64, gorgonia.WithShape(7), gorgonia.WithName("x"))

	// Update Hidden layer input shape to 7
	w0 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(7, 16), gorgonia.WithName("w0"), gorgonia.WithInit(gorgonia.GlorotU(multiplier)))
	w1 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(16, 5), gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.GlorotU(multiplier)))

	l0 := gorgonia.Must(gorgonia.Rectify(gorgonia.Must(gorgonia.Mul(x, w0))))
	out := gorgonia.Must(gorgonia.Tanh(gorgonia.Must(gorgonia.Mul(l0, w1))))

	vm := gorgonia.NewTapeMachine(g)
	return &Brain{g: g, w0: w0, w1: w1, x: x, out: out, vm: vm}
}

func (b *Brain) PredictSpatial(input []float64) (action int, latNudge, lonNudge float64) {
	gorgonia.Let(b.x, tensor.New(tensor.WithBacking(input)))
	b.vm.RunAll()
	res := b.out.Value().Data().([]float64)
	b.vm.Reset()

	// 1. Determine Action (Argmax of the first 3 nodes)
	action = 0
	maxVal := res[0]
	for i := 1; i < 3; i++ {
		if res[i] > maxVal {
			action = i
			maxVal = res[i]
		}
	}

	// 2. Extract Spatial Nudges (Nodes 3 and 4)
	latNudge = res[3]
	lonNudge = res[4]

	return action, latNudge, lonNudge
}

func (b *Brain) Predict(input []float64) int {
	gorgonia.Let(b.x, tensor.New(tensor.WithBacking(input)))
	b.vm.RunAll()
	res := b.out.Value().Data().([]float64)
	b.vm.Reset()
	maxIdx := 0
	for i, v := range res {
		if v > res[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}

func loadSystemState() {
	// 1. Load Radar Positions
	if data, err := os.ReadFile(RadarFile); err == nil {
		var savedEntities []Entity
		if err := json.Unmarshal(data, &savedEntities); err == nil {
			mu.Lock()
			for _, e := range savedEntities {
				if e.Type == "RADAR" {
					entities[e.ID] = &Entity{
						ID: e.ID, Type: e.Type,
						Lat: e.Lat, Lon: e.Lon,
						LastMoved: e.LastMoved,
					}
					kills[e.ID] = 0
				}
			}
			mu.Unlock()
		}
	}

	// 2. Load NN and Scaling Benchmarks
	f, err := os.Open(BrainFile)
	if err != nil {
		return
	}
	defer f.Close()

	dec := gob.NewDecoder(f)
	var w0T, w1T *tensor.Dense

	if err := dec.Decode(&w0T); err == nil {
		gorgonia.Let(brain.w0, w0T)
	}
	if err := dec.Decode(&w1T); err == nil {
		gorgonia.Let(brain.w1, w1T)
	}

	dec.Decode(&minEverRadars)
	dec.Decode(&MaxRadars) // Recover the target scale
}

func getTetheredCoords() (float64, float64) {
	city := entities[cityNames[rand.Intn(len(cityNames))]]
	lat := city.Lat + (rand.Float64()-0.5)*14.0
	lon := city.Lon + (rand.Float64()-0.5)*14.0
	if getDistanceKM(city.Lat, city.Lon, lat, lon) > TetherRadiusKM {
		return city.Lat, city.Lon
	}
	return lat, lon
}

func saveSystemState() {
	mu.RLock()
	defer mu.RUnlock()

	// 1. Save NN Weights, Min-Ever Record, and Scaled Limit
	f, err := os.Create(BrainFile)
	if err == nil {
		enc := gob.NewEncoder(f)
		enc.Encode(brain.w0.Value())
		enc.Encode(brain.w1.Value())
		enc.Encode(minEverRadars)
		enc.Encode(MaxRadars) // Persist the scaling limit
		f.Close()
	}

	// 2. Save Optimized Radar Positions
	var optimizedFleet []Entity
	for _, e := range entities {
		if e.Type == "CITY" || e.Type == "RADAR" {
			optimizedFleet = append(optimizedFleet, *e)
		}
	}

	jsonData, err := json.MarshalIndent(optimizedFleet, "", "  ")
	if err == nil {
		os.WriteFile(RadarFile, jsonData, 0644)
	}
}

func autonomousEraReset() {
	mu.Lock()
	defer mu.Unlock()

	var radarIDs []string
	for id, e := range entities {
		if e.Type == "RADAR" {
			radarIDs = append(radarIDs, id)
		}
	}
	rCount := len(radarIDs)

	// 1. RECOVERY SEEDING
	if rCount < MinRadars || budget < 0 {
		fmt.Printf("\n[!] RECOVERING: Seeding fleet...\n")
		for i := 0; i < MinRadars; i++ {
			id := fmt.Sprintf("R-SEED-%d-%d", currentCycle, i)
			lat, lon := getTetheredCoords()
			entities[id] = &Entity{ID: id, Type: "RADAR", Lat: lat, Lon: lon}
		}
		radarIDs = nil
		for id, e := range entities {
			if e.Type == "RADAR" {
				radarIDs = append(radarIDs, id)
			}
		}
		rCount = len(radarIDs)
	}

	// 2. TARGET SCALING (The New Logic)
	// If we hit 99% success, we tighten the noose on the AI.
	if successRate >= 99.0 && totalThreats >= 50 {
		if rCount < minEverRadars {
			minEverRadars = rCount

			// TARGET SCALING: Lower the global limit to the new record.
			// This prevents the AI from ever "buying back" into inefficiency.
			MaxRadars = minEverRadars

			fmt.Printf("\nTARGET SCALED: Max fleet size is now %d units.\n", MaxRadars)
			saveSystemState()
		}
	}

	// 3. BANKRUPTCY HANDLING
	if budget < -BankruptcyLimit {
		fmt.Println("\nCRISIS: Applying Soft Gaussian Mutation...")
		brain.Mutate(0.1) // 10% jitter to current knowledge
		budget = 500000.0
		goto FinalizeReset
	}

	// 4. AGGRESSIVE PRUNING
	if successRate >= 95.0 {
		sort.Slice(radarIDs, func(i, j int) bool {
			return kills[radarIDs[i]] < kills[radarIDs[j]]
		})

		// Drop 15% of the current fleet
		numToDrop := int(float64(rCount) * 0.15)
		dropped := 0
		for i := 0; i < numToDrop; i++ {
			if len(radarIDs)-dropped <= MinRadars {
				break
			}
			id := radarIDs[i]
			delete(entities, id)
			delete(kills, id)
			dropped++

			// Efficiency Bonus rewards smaller fleet sizes
			scalingFactor := 1.0 + (100.0 / float64(len(entities)-len(cityNames)+1))
			budget += EfficiencyBonus * scalingFactor
		}

		// Benchmark Mode: Calculate KM2 Efficiency
		// Area of circle = PI * r^2.
		radarArea := math.Pi * math.Pow(RadarRadiusKM, 2)
		theoreticalCoverage := float64(len(radarIDs)-dropped) * radarArea
		rCount = rCount - dropped
		effectiveArea := theoreticalCoverage * (successRate / 100.0) // Adjusted for actual hits

		fmt.Printf("PERFORMANCE: %.2f Million KM2 of ACTIVE protection.\n", effectiveArea/1e6)

		fmt.Printf("[+] ERA %d: Success %.1f%% | Limit: %d | Dropped: %d\n", currentCycle, successRate, MaxRadars, dropped)
		saveSystemState()
	}

	for id := range kills {
		kills[id] = 0
	}

FinalizeReset:
	currentCycle++
	eraStartTime = simClock
	if budget < 500000.0 {
		budget = 500000.0
	}
	totalEraSpending = 0
	totalThreats = 0
	totalIntercepts = 0
	for i := range quadrantMisses {
		quadrantMisses[i] = 0
	}
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
		if simClock.IsZero() {
			simClock = time.Now()
			eraStartTime = time.Now()
		}

		// Handle Era Transitions
		if simClock.After(eraStartTime.Add(EraDuration)) || forceReset {
			forceReset = false
			mu.Unlock()
			autonomousEraReset()
			continue
		}

		// --- 1,000 THREAT BATCH ---
		const batchSize = 1000
		for b := 0; b < batchSize; b++ {
			totalThreats++
			target := entities[cityNames[rand.Intn(len(cityNames))]]
			intercepted := false

			// Check for interception
			for id, e := range entities {
				if e.Type == "RADAR" && getDistanceKM(e.Lat, e.Lon, target.Lat, target.Lon) < RadarRadiusKM {
					kills[id]++
					budget += InterceptReward
					totalIntercepts++
					intercepted = true
					break
				}
			}

			if !intercepted {
				budget -= CityImpactPenalty
				// SPATIAL MEMORY: Log which quadrant the miss occurred in
				// Indexing: 0:NW, 1:NE, 2:SW, 3:SE
				qIdx := 0
				if target.Lat < 0 {
					qIdx += 2
				} // South
				if target.Lon > 0 {
					qIdx += 1
				} // East
				quadrantMisses[qIdx]++
			}
			simClock = simClock.Add(4 * time.Hour)
		}

		// Calculate current performance metrics
		if totalThreats > 0 {
			successRate = (float64(totalIntercepts) / float64(totalThreats)) * 100
		}

		rCount := 0
		for _, e := range entities {
			if e.Type == "RADAR" {
				rCount++
			}
		}

		// --- AI SPATIAL LOGIC (7-Input Vector) ---
		// We pass Budget, Density, Success, and the 4 Quadrant Miss-counts
		input := []float64{
			math.Max(-1, math.Min(budget/BankruptcyLimit, 1.0)),
			float64(rCount) / float64(MaxRadars),
			successRate / 100.0,
			math.Min(quadrantMisses[0]/100, 1.0), // Normalized NW Misses
			math.Min(quadrantMisses[1]/100, 1.0), // Normalized NE Misses
			math.Min(quadrantMisses[2]/100, 1.0), // Normalized SW Misses
			math.Min(quadrantMisses[3]/100, 1.0), // Normalized SE Misses
		}

		action, latNudge, lonNudge := brain.PredictSpatial(input)

		// Calculate a dynamic multiplier based on failure
		// Low success = big jumps (up to 20x); High success = tiny adjustments (min 1x)
		precisionScale := 20.0 * (1.0 - (successRate / 100.0))
		if precisionScale < 1.0 {
			precisionScale = 1.0
		}

		switch action {
		case 1: // BUILD NEW
			// EMERGENCY OVERRIDE: Allow building if below MinRadars regardless of budget
			if (budget >= RadarCost || rCount < MinRadars) && rCount < MaxRadars {
				baseLat, baseLon := getTetheredCoords()
				newLat := baseLat + (latNudge * precisionScale * 2.0) // Aggressive search
				newLon := baseLon + (lonNudge * precisionScale * 2.0)
				id := fmt.Sprintf("R-AI-%d-%d", currentCycle, rand.Intn(1e7))
				entities[id] = &Entity{ID: id, Type: "RADAR", Lat: newLat, Lon: newLon}
				budget -= RadarCost
			}

		case 2: // RELOCATE WORST PERFORMER
			if budget >= RelocationCost && rCount > 0 {
				var worstID string
				minKills := 999999
				for id, count := range kills {
					if entities[id].Type == "RADAR" && count < minKills {
						minKills = count
						worstID = id
					}
				}
				if e, ok := entities[worstID]; ok {
					baseLat, baseLon := getTetheredCoords()
					// Relocation uses raw precision scale for surgical placement
					e.Lat = baseLat + (latNudge * precisionScale)
					e.Lon = baseLon + (lonNudge * precisionScale)
					e.LastMoved = time.Now().UnixMilli()
					kills[worstID] = 0
					budget -= RelocationCost
				}
			}
		}
		mu.Unlock()

		// Breather for the HTTP scheduler to prevent UI update starvation
		time.Sleep(1 * time.Millisecond)
	}
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	cityData := map[string][]float64{
		// Canada
		"Toronto": {43.65, -79.38}, "Montreal": {45.50, -73.56}, "Vancouver": {49.28, -123.12},
		"Calgary": {51.04, -114.07}, "Edmonton": {53.54, -113.49}, "Ottawa": {45.42, -75.69},
		"Winnipeg": {49.89, -97.13}, "Quebec City": {46.81, -71.20}, "Halifax": {44.64, -63.57},
		"Victoria": {48.42, -123.36}, "Saskatoon": {52.13, -106.67}, "St. John's": {47.56, -52.71},
		"Yellowknife": {62.45, -114.37}, "Whitehorse": {60.72, -135.05},

		// USA
		"New York": {40.71, -74.00}, "Los Angeles": {34.05, -118.24}, "Chicago": {41.87, -87.62},
		"Houston": {29.76, -95.36}, "Phoenix": {33.45, -112.07}, "Philadelphia": {39.95, -75.16},
		"San Francisco": {37.77, -122.42}, "Seattle": {47.61, -122.33},
		"Washington DC": {38.89, -77.03}, "Miami": {25.76, -80.19},
		"Boston": {42.36, -71.05}, "Atlanta": {33.75, -84.39},
		"Denver": {39.74, -104.99}, "Las Vegas": {36.17, -115.14},
		"San Diego": {32.72, -117.16}, "Dallas": {32.78, -96.80},
		"Anchorage": {61.21, -149.90}, "Honolulu": {21.30, -157.85},

		// Mexico & Central America
		"Mexico City": {19.43, -99.13}, "Guadalajara": {20.67, -103.35},
		"Monterrey": {25.68, -100.31}, "Panama City": {8.98, -79.52},
		"San Jose": {9.93, -84.08},

		// South America
		"Sao Paulo": {-23.55, -46.63}, "Rio de Janeiro": {-22.90, -43.17},
		"Buenos Aires": {-34.60, -58.38}, "Cordoba": {-31.42, -64.19},
		"Lima": {-12.04, -77.04}, "Bogota": {4.71, -74.07},
		"Santiago": {-33.44, -70.66}, "Caracas": {10.48, -66.90},
		"Montevideo": {-34.90, -56.16}, "Asuncion": {-25.26, -57.58},
		"Quito": {-0.18, -78.47}, "La Paz": {-16.50, -68.15},

		// Europe
		"London": {51.50, -0.12}, "Paris": {48.85, 2.35}, "Berlin": {52.52, 13.40},
		"Rome": {41.90, 12.49}, "Madrid": {40.41, -3.70}, "Barcelona": {41.38, 2.17},
		"Lisbon": {38.72, -9.14}, "Amsterdam": {52.37, 4.90},
		"Brussels": {50.85, 4.35}, "Vienna": {48.21, 16.37},
		"Prague": {50.08, 14.43}, "Budapest": {47.50, 19.04},
		"Warsaw": {52.22, 21.01}, "Stockholm": {59.32, 18.06},
		"Oslo": {59.91, 10.75}, "Copenhagen": {55.68, 12.57},
		"Helsinki": {60.17, 24.94}, "Dublin": {53.35, -6.26},
		"Zurich": {47.38, 8.54}, "Geneva": {46.20, 6.15},
		"Athens": {37.98, 23.73}, "Istanbul": {41.00, 28.97},
		"Kyiv": {50.45, 30.52}, "Moscow": {55.75, 37.61},

		// Middle East
		"Tel Aviv": {32.08, 34.78}, "Jerusalem": {31.77, 35.21},
		"Dubai": {25.20, 55.27}, "Abu Dhabi": {24.45, 54.38},
		"Riyadh": {24.71, 46.67}, "Jeddah": {21.49, 39.19},
		"Doha": {25.29, 51.53}, "Kuwait City": {29.38, 47.98},
		"Tehran": {35.68, 51.38}, "Baghdad": {33.31, 44.36},
		"Amman": {31.95, 35.93},

		// Africa
		"Cairo": {30.04, 31.23}, "Alexandria": {31.20, 29.92},
		"Lagos": {6.52, 3.37}, "Abuja": {9.07, 7.48},
		"Johannesburg": {-26.20, 28.04}, "Cape Town": {-33.92, 18.42},
		"Pretoria": {-25.75, 28.19}, "Nairobi": {-1.29, 36.82},
		"Addis Ababa": {8.98, 38.79}, "Accra": {5.56, -0.20},
		"Casablanca": {33.57, -7.58}, "Rabat": {34.02, -6.83},
		"Tunis": {36.81, 10.18}, "Algiers": {36.75, 3.06},

		// Asia
		"Beijing": {39.90, 116.40}, "Shanghai": {31.23, 121.47},
		"Shenzhen": {22.54, 114.06}, "Guangzhou": {23.13, 113.26},
		"Hong Kong": {22.31, 114.16}, "Taipei": {25.03, 121.56},
		"Tokyo": {35.68, 139.65}, "Osaka": {34.69, 135.50},
		"Seoul": {37.56, 126.97}, "Busan": {35.18, 129.07},
		"Bangkok": {13.75, 100.50}, "Hanoi": {21.03, 105.85},
		"Ho Chi Minh City": {10.82, 106.63},
		"Singapore":        {1.35, 103.81}, "Kuala Lumpur": {3.14, 101.69},
		"Jakarta": {-6.20, 106.81}, "Manila": {14.59, 120.98},
		"Mumbai": {19.07, 72.87}, "New Delhi": {28.61, 77.20},
		"Bangalore": {12.97, 77.59}, "Chennai": {13.08, 80.27},
		"Karachi": {24.86, 67.01}, "Lahore": {31.55, 74.34},
		"Dhaka":  {23.81, 90.41},
		"Astana": {51.16, 71.47}, "Almaty": {43.24, 76.88},

		// Oceania
		"Sydney": {-33.86, 151.20}, "Melbourne": {-37.81, 144.96},
		"Brisbane": {-27.47, 153.03}, "Perth": {-31.95, 115.86},
		"Auckland": {-36.84, 174.76}, "Wellington": {-41.29, 174.78},
	}

	for name, pos := range cityData {
		entities[name] = &Entity{ID: name, Type: "CITY", Lat: pos[0], Lon: pos[1]}
		cityNames = append(cityNames, name)
	}
	setupSimulation()

	// FORCED INITIALIZATION: Ensure the NN has assets to work with immediately.
	mu.Lock()
	radarCount := 0
	for _, e := range entities {
		if e.Type == "RADAR" {
			radarCount++
		}
	}
	if radarCount < MinRadars {
		fmt.Println("INITIALIZING SEED FLEET...")
		for i := 0; i < MinRadars; i++ {
			id := fmt.Sprintf("R-START-%d", i)
			lat, lon := getTetheredCoords()
			entities[id] = &Entity{ID: id, Type: "RADAR", Lat: lat, Lon: lon}
		}
	}
	mu.Unlock()
	wallStart = time.Now()

	go runPhysicsEngine()

	http.HandleFunc("/panic", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		brain = NewBrain(mutationRate)
		budget = 500000.0                   // Restore starting capital
		entities = make(map[string]*Entity) // Clear the failed 0-radar state
		for name, pos := range cityData {
			entities[name] = &Entity{ID: name, Type: "CITY", Lat: pos[0], Lon: pos[1]}
		}
		forceReset = true
		mu.Unlock()
		w.Write([]byte("Shield Re-initialized"))
	})
	http.HandleFunc("/intel", func(w http.ResponseWriter, r *http.Request) {
		mu.RLock()
		defer mu.RUnlock()

		// Calculate Years Per Second
		realSeconds := time.Since(wallStart).Seconds()
		simHours := simClock.Sub(eraStartTime).Hours() + (float64(currentCycle-1) * EraDuration.Hours())
		simYears := simHours / (24 * 365)

		yps := 0.0
		if realSeconds > 0 {
			yps = simYears / realSeconds
		}

		var all []Entity
		for _, e := range entities {
			all = append(all, *e)
		}
		json.NewEncoder(w).Encode(map[string]interface{}{
			"cycle":    currentCycle,
			"budget":   budget,
			"entities": all,
			"success":  successRate,
			"min_ever": minEverRadars, // Missing in your current version
			"yps":      yps,           // You can calculate this as wall-time vs sim-time
		})
	})
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		template.Must(template.New("v").Parse(uiHTML)).Execute(w, nil)
	})
	fmt.Println("AEGIS BENCHMARK RUNNING AT :8080")
	http.ListenAndServe(":8080", nil)
}

func setupSimulation() {
	brain = NewBrain(mutationRate) // Initialize graph first
	loadSystemState()              // Overlay persistent data
}

const uiHTML = `
<!DOCTYPE html><html><head><title>AEGIS AI OPTIMIZER v2</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
    body { margin:0; background:#000; color:#0f0; font-family:monospace; overflow:hidden; } 
    #stats { 
        position:fixed; top:10px; left:10px; z-index:1000; 
        background:rgba(0,10,20,0.9); padding:15px; border:1px solid #0af; 
        box-shadow: 0 0 10px #0af; line-height:1.6; min-width: 220px;
    } 
    #map { height:100vh; width:100vw; background: #000; }
    .stat-row { display: flex; justify-content: space-between; border-bottom: 1px solid #0af3; }
    .stat-val { color: #fff; font-weight: bold; }
    .record-val { color: #f0f; font-weight: bold; }
    .legend { font-size: 0.8em; margin-top: 10px; color: #aaa; }
</style></head>
<body>
<div id="stats">
    <div class="stat-row"><span>ERA:</span> <span id="era" class="stat-val">0</span></div>
    <div class="stat-row"><span>RADARS:</span> <span><span id="rcount" class="stat-val">0</span> / <span id="maxr" class="record-val">0</span></span></div>
    <div class="stat-row"><span>SUCCESS:</span> <span id="success" class="stat-val">0</span>%</div>
    <div class="stat-row"><span>BEST:</span> <span id="min_ever" class="record-val">0</span></div>
    <div class="stat-row"><span>BUDGET:</span> $<span id="budget" class="stat-val">0</span></div>
    <div class="stat-row"><span>SPEED:</span> <span id="yps" class="stat-val">0</span> Y/sec</div>
    <div class="legend">
        <span style="color:#0f0">●</span> Coverage <span style="color:#f00">●</span> Target <span style="color:#ff0">●</span> Moving
    </div>
</div>
<div id="map"></div>
<script>
    var map = L.map('map', {zoomControl:false, attributionControl:false}).setView([20, 0], 2);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png').addTo(map);
    
    var layers = {};
    var isSyncing = false;

    async function sync() {
        if (isSyncing) return;
        isSyncing = true;

        try {
            const r = await fetch('/intel'); 
            if (!r.ok) throw new Error('Backend Starved');
            const d = await r.json();

            // 1. Update UI Elements
            document.getElementById('era').innerText = d.cycle;
            document.getElementById('success').innerText = (d.success || 0).toFixed(1);
            document.getElementById('budget').innerText = Math.floor(d.budget).toLocaleString();
            document.getElementById('min_ever').innerText = d.min_ever;
            document.getElementById('maxr').innerText = d.min_ever;
            document.getElementById('yps').innerText = (d.yps || 0).toFixed(2);

            const entities = d.entities || [];
            const currentIds = new Set(entities.map(e => e.id));
            const now = Date.now();

            // 2. Prune Dead Layers (Optimized)
            for (let id in layers) {
                if (!currentIds.has(id)) {
                    map.removeLayer(layers[id]);
                    delete layers[id];
                }
            }
            
            let radarCount = 0;
            entities.forEach(e => {
                if (!layers[e.id]) {
                    // Create New
                    if (e.type === 'RADAR') {
                        layers[e.id] = L.circle([e.lat, e.lon], {
                            radius: 1200000, color: '#0f0', weight: 0.5, fillOpacity: 0.1
                        }).addTo(map);
                    } else {
                        layers[e.id] = L.circleMarker([e.lat, e.lon], {
                            radius: 3, color: '#f00', weight: 1, fillOpacity: 0.5
                        }).addTo(map);
                    }
                } else {
                    // Update Existing (Throttled Move)
                    layers[e.id].setLatLng([e.lat, e.lon]);
                    
                    // Flash Logic for Relocation
                    if (e.last_moved && (now - e.last_moved < 1200)) {
                        layers[e.id].setStyle({color: '#ff0', weight: 3, fillOpacity: 0.5});
                    } else if (e.type === 'RADAR') {
                        layers[e.id].setStyle({color: '#0f0', weight: 0.5, fillOpacity: 0.1});
                    }
                }
                if (e.type === 'RADAR') radarCount++;
            });

            document.getElementById('rcount').innerText = radarCount;

        } catch (err) {
            console.warn("Sync throttled:", err.message);
        } finally {
            isSyncing = false;
            // Recursive timeout prevents request overlap
            setTimeout(sync, 400); 
        }
    }

    sync(); // Boot up the loop
</script></body></html>`

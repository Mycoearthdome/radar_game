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
	RadarRadiusKM       = 1200.0
	TetherRadiusKM      = 800.0
	RadarCost           = 250000.0
	RelocationCost      = 5000.0
	EnforcementFee      = 5000000.0
	MinRadars           = 60
	CityImpactPenalty   = 200000.0
	InterceptReward     = 10000.0
	EarthRadius         = 6371.0
	EraDuration         = 24 * time.Hour
	TimeStepping        = 20 * time.Minute
	RadarFile           = "RADAR.json"
	BrainFile           = "BRAIN.gob"
	BankruptcyLimit     = 0.0
	GridSize            = 10 // 10 degree cells
	Cols                = 36 // 360 / 10
	Rows                = 18 // 180 / 10
	TargetSuccess       = 100.0
	RequiredWinStreak   = 5000   // Number of eras to maintain 100% before "winning"
	MissileMaxSpeedKMH  = 8000.0 // Hypersonic Mach 6.5+
	MissileGMLimit      = 40.0   // Max turning force
	ProximityRadiusKM   = 5.0    // Blast zone
	KineticRadiusKM     = 0.1    // Direct hit
	BaseKillProbability = 0.85   // Success rate for proximity detonation
	MaxEnergy           = 100.0  // Starting energy percentage
	DragPenaltyBase     = 0.05   // Energy loss per KM of flight
	ManeuverEnergyCost  = 2.5    // Extra cost for high-G turns
	MaxSatellites       = 20
	SatelliteRangeKM    = 2500.0                  // Higher vantage point = wider reach
	LaunchInterval      = 1 * 24 * 31 * time.Hour //one month
	MissileMaxRangeKM   = 2500.0                  // The absolute maximum distance a missile can travel
	FuelConsumption     = 0.04                    // Energy/Fuel cost per KM traveled by the threat
	StartBudget         = 1000000000.0
	EmergencyBudget     = StartBudget
)

var (
	entities             = make(map[string]*Entity)
	kills                = make(map[string]int)
	budget               = StartBudget
	totalEraSpending     = 0.0
	mu                   sync.RWMutex
	eraStartTime         time.Time
	simClock             time.Time
	wallStart            time.Time // For Benchmark
	currentCycle         = 1
	brain                *Brain
	successRate          float64
	lastEraSuccess       float64
	totalThreats         int
	totalIntercepts      int
	MaxRadars            = 500
	minEverRadars        = MaxRadars
	forceReset           = false
	cityNames            []string
	mutationRate         = 0.01               //was 1.0
	quadrantMisses       = make([]float64, 4) // [NW, NE, SW, SE]
	winStreakCounter     = 0
	isSimulationOver     = false
	EfficiencyBonus      = 100000.0
	difficultyMultiplier = 1.0
)

var launchSites = []struct{ Lat, Lon float64 }{
	{28.57, -80.64},  // Cape Canaveral
	{34.74, -120.57}, // Vandenberg
	{5.23, -52.76},   // Kourou
	{45.96, 63.30},   // Baikonur
	{18.65, 100.48},  // Jiuquan
}

type Satellite struct {
	ID        string
	LaunchLat float64
	LaunchLon float64
	StartTime int64
}

type Entity struct {
	ID        string  `json:"id"`
	Type      string  `json:"type"`
	Lat       float64 `json:"lat"`
	Lon       float64 `json:"lon"`
	LaunchLat float64 `json:"launch_lat"`
	LaunchLon float64 `json:"launch_lon"`
	StartTime int64   `json:"start_time"`
	LastMoved int64   `json:"last_moved"`
}

type Brain struct {
	g   *gorgonia.ExprGraph
	w0  *gorgonia.Node
	w1  *gorgonia.Node
	x   *gorgonia.Node // The input placeholder
	out *gorgonia.Node // The output (Tanh) node
}

func (b *Brain) Mutate(rate float64) {
	// Access the underlying tensors for weights
	w0Data := b.w0.Value().Data().([]float64)
	w1Data := b.w1.Value().Data().([]float64)

	// Apply improved Gaussian noise + scaling to Hidden Layer
	for i := range w0Data {
		if rand.Float64() < 0.9 {
			w0Data[i] += rand.NormFloat64() * rate
		} else {
			// Helps the AI escape "local minima" by scaling the weight
			w0Data[i] *= (1.0 + (rand.NormFloat64() * rate))
		}
	}

	// Apply improved Gaussian noise + scaling to Output Layer
	for i := range w1Data {
		if rand.Float64() < 0.9 {
			w1Data[i] += rand.NormFloat64() * rate
		} else {
			w1Data[i] *= (1.0 + (rand.NormFloat64() * rate))
		}
	}
}

func (b *Brain) AdaptiveMutate(success float64) {
	// 1. Calculate Dynamic Jitter Rate
	// High success (99%) = tiny jitter (0.002) - fine-tuning only.
	// Low success (50%) = large jitter (0.1) - seeking new strategies.
	rate := 0.2 * (1.0 - (success / 100.0))

	// Clamp the rate to avoid total brain-wipe
	if rate < 0.002 {
		rate = 0.002
	}
	mutationRate = rate

	fmt.Printf("[Mutation] Success: %.2f%% | Applying Jitter Rate: %.5f\n", success, rate)

	// 2. Access the Raw Data of the weights
	// We mutate both the input-to-hidden (w0) and hidden-to-output (w1) layers.
	for _, w := range []*gorgonia.Node{b.w0, b.w1} {
		wT := w.Value().Data().([]float64)

		for i := range wT {
			// BIAS TOWARD STABILITY:
			// Only mutate a percentage of weights based on the success rate.
			// If success is 90%, only 10% of weights get jittered.
			if rand.Float64() > (success / 100.0) {
				// Apply Gaussian jitter
				jitter := rand.NormFloat64() * rate
				wT[i] += jitter
			}
		}
	}
}

// Helper to get grid ID from coordinates
func getGridID(lat, lon float64) int {
	row := int((lat + 90) / GridSize)
	col := int((lon + 180) / GridSize)
	return row*Cols + col
}

func NewBrain(multiplier float64) *Brain {
	g := gorgonia.NewGraph()

	// Define the Input Placeholder (This is what was nil)
	x := gorgonia.NewMatrix(g, tensor.Float64,
		gorgonia.WithShape(1, 16),
		gorgonia.WithName("x"))

	// Define Weights
	w0 := gorgonia.NewMatrix(g, tensor.Float64,
		gorgonia.WithShape(16, 24),
		gorgonia.WithName("w0"),
		gorgonia.WithInit(gorgonia.GlorotU(multiplier)))

	w1 := gorgonia.NewMatrix(g, tensor.Float64,
		gorgonia.WithShape(24, 3),
		gorgonia.WithName("w1"),
		gorgonia.WithInit(gorgonia.GlorotU(multiplier)))

	// --- Define the Graph Path ---
	// Layer 1
	h0 := gorgonia.Must(gorgonia.Mul(x, w0))
	a0 := gorgonia.Must(gorgonia.LeakyRelu(h0, 0.1))

	// Layer 2 (Output)
	h1 := gorgonia.Must(gorgonia.Mul(a0, w1))
	out := gorgonia.Must(gorgonia.Tanh(h1))

	return &Brain{
		g:   g,
		w0:  w0,
		w1:  w1,
		x:   x,   // Store the reference for Let()
		out: out, // Store the reference for reading results
	}
}

func (b *Brain) PredictSpatial(inputs []float64) (int, float64, float64) {
	// 1. Safety Check: Ensure the input vector matches our new 16-node architecture
	if len(inputs) != 16 {
		fmt.Printf("[ERROR] Expected 16 inputs, got %d\n", len(inputs))
		return 0, 0, 0
	}

	// 2. Create input tensor
	// We use a 1x16 shape to match our Matrix multiplication requirements (1x16 * 16x24)
	inputT := tensor.New(tensor.WithShape(1, 16), tensor.WithBacking(inputs))

	// 3. Bind the tensor to the graph's input node
	// Note: 'b.x' must be defined in your Brain struct as the input placeholder node
	err := gorgonia.Let(b.x, inputT)
	if err != nil {
		return 0, 0, 0
	}

	// 4. Execution using TapeMachine
	// We use a persistent VM if possible, or create a light one for this pass
	vm := gorgonia.NewTapeMachine(b.g)
	defer vm.Close() // Ensures no memory leaks during high-speed eras

	if err := vm.RunAll(); err != nil {
		return 0, 0, 0
	}

	// 5. Extract results from the 'out' node
	// 'b.out' is the final Tanh node in your graph
	res := b.out.Value().Data().([]float64)

	// 6. Strategic Action Mapping
	// res[0] is the Action Selector (Tanh: -1 to 1)
	action := 0
	if res[0] > 0.4 {
		action = 1 // BUILD: Decisive positive signal
	} else if res[0] < -0.4 {
		action = 2 // RELOCATE: Decisive negative signal
	}
	// Middle ground (-0.4 to 0.4) is STAY/OBSERVE

	// 7. Return results
	// res[1] = Latitudinal Nudge
	// res[2] = Longitudinal Nudge
	return action, res[1], res[2]
}

func loadSystemState() {
	if data, err := os.ReadFile(RadarFile); err == nil {
		var savedEntities []Entity
		if err := json.Unmarshal(data, &savedEntities); err == nil {
			mu.Lock()
			// Reset entities to clear any seeded data from main()
			entities = make(map[string]*Entity)

			for _, e := range savedEntities {
				newE := &Entity{
					ID: e.ID, Type: e.Type,
					Lat: e.Lat, Lon: e.Lon,
					LaunchLat: e.LaunchLat, LaunchLon: e.LaunchLon,
					StartTime: e.StartTime, LastMoved: e.LastMoved,
				}

				// FIX 1: Sync Satellite Orbital Clocks
				// If we are loading a satellite, we must ensure its StartTime
				// relative to the current simClock is preserved or reset
				// to prevent orbital jumping.
				if e.Type == "SAT" && simClock.IsZero() == false {
					newE.StartTime = simClock.Unix()
				}

				entities[e.ID] = newE

				// Re-initialize kills map for radars
				if e.Type == "RADAR" {
					kills[e.ID] = 0
				}
			}
			mu.Unlock()
			fmt.Printf("Successfully loaded %d entities from persistence.\n", len(savedEntities))
		}
	}

	f, err := os.Open(BrainFile)
	if err != nil {
		return
	}
	defer f.Close()

	dec := gob.NewDecoder(f)
	var w0T, w1T *tensor.Dense

	// FIX 2: Correct Input Migration Logic
	if err := dec.Decode(&w0T); err == nil {
		currentShape := w0T.Shape()[0]
		// Migration logic for 12-input architecture
		if currentShape < 12 {
			fmt.Printf("Migrating %d-input brain to 12-input...\n", currentShape)
			newW0 := tensor.New(tensor.WithShape(12, 16), tensor.WithBacking(make([]float64, 12*16)))
			oldData := w0T.Data().([]float64)
			newData := newW0.Data().([]float64)

			// Copy old weights into the new larger tensor
			copy(newData, oldData)

			// Initialize the new "knowledge" rows (Satellite and Density awareness)
			// using a small jitter so the AI starts with neutral curiosity.
			for i := len(oldData); i < 192; i++ {
				newData[i] = (rand.Float64() - 0.5) * 0.01
			}
			gorgonia.Let(brain.w0, newW0)
		} else {
			gorgonia.Let(brain.w0, w0T)
		}
	}

	if err := dec.Decode(&w1T); err == nil {
		gorgonia.Let(brain.w1, w1T)
	}

	// Ensure we preserve the historical "Tightest Fleet" record
	dec.Decode(&minEverRadars)
	dec.Decode(&MaxRadars)
}

func saveSystemState() {
	// 1. Save NN Weights, Min-Ever Record, and Scaled Limit
	// We use GOB because it preserves the tensor.Dense structures perfectly
	f, err := os.Create(BrainFile)
	if err == nil {
		enc := gob.NewEncoder(f)

		// Encode weights (Note: Value() returns the underlying tensor)
		enc.Encode(brain.w0.Value())
		enc.Encode(brain.w1.Value())

		// Encode metrics for the NN to maintain its current "Era" difficulty
		enc.Encode(minEverRadars)
		enc.Encode(MaxRadars)

		f.Close()
	} else {
		fmt.Printf("Error saving brain state: %v\n", err)
	}

	// 2. Save Optimized Entity Positions (Radars & Satellites)
	// We save these to JSON so they can be reloaded as a persistent fleet
	var optimizedFleet []Entity

	mu.Lock()
	for _, e := range entities {
		// Only save persistent types. Cities are usually static,
		// but we save them to maintain the map context.
		if e.Type == "CITY" || e.Type == "RADAR" || e.Type == "SAT" {
			optimizedFleet = append(optimizedFleet, *e)
		}
	}
	mu.Unlock()

	jsonData, err := json.MarshalIndent(optimizedFleet, "", "  ")
	if err == nil {
		err = os.WriteFile(RadarFile, jsonData, 0644)
		if err != nil {
			fmt.Printf("Error writing Radar JSON: %v\n", err)
		}
	} else {
		fmt.Printf("Error marshaling entity data: %v\n", err)
	}
}

func autonomousEraReset() {
	mu.Lock()
	defer mu.Unlock()

	// 1. EVALUATE PERFORMANCE & BONUSES
	// Reward scaling based on efficiency tiers
	if successRate >= 100.0 {
		difficultyMultiplier += 0.1 // Threats get 10% faster every era the AI wins
		EfficiencyBonus = EfficiencyBonus * 1.1
	} else if successRate >= 99.0 {
		difficultyMultiplier += 0.05 // Threats get 5% faster every era the AI wins
		EfficiencyBonus = EfficiencyBonus * 1.05
	} else if successRate >= 99.5 {
		EfficiencyBonus = EfficiencyBonus * 1.025
	} else {
		EfficiencyBonus = 100000.0 // Reset to baseline on failure
	}

	// 2. WIN STREAK & CONVERGENCE
	// Ensure a minimum threat volume before counting a 'Win' to prevent cheesing
	if successRate >= 100.0 && (totalThreats+totalIntercepts) > 50 {
		winStreakCounter++
		fmt.Printf("\n[!] VICTORY STREAK: %d/%d Eras at 100%% efficiency.\n", winStreakCounter, RequiredWinStreak)

		if winStreakCounter >= RequiredWinStreak {
			fmt.Println("\n==================================================")
			fmt.Println("CONVERGENCE REACHED: AEGIS SHIELD IS OPTIMIZED")
			fmt.Printf("Final Fleet Size: %d | Total Eras: %d\n", len(entities)-len(cityNames), currentCycle)
			fmt.Println("==================================================")

			isSimulationOver = true
			saveSystemState()
			go func() {
				time.Sleep(2 * time.Second)
				os.Exit(0)
			}()
			return
		}
	} else {
		if winStreakCounter > 0 {
			fmt.Printf("\n[!] STREAK BROKEN: Efficiency fell to %.2f%%\n", successRate)
		}
		winStreakCounter = 0
	}

	// 3. TARGET SCALING (Fleet Tightening)
	radarIDs := []string{}
	for id, e := range entities {
		if e.Type == "RADAR" {
			radarIDs = append(radarIDs, id)
		}
	}
	rCount := len(radarIDs)

	if successRate >= 99.5 && (totalThreats+totalIntercepts) >= 50 {
		if rCount < minEverRadars {
			minEverRadars = rCount
			MaxRadars = minEverRadars
			fmt.Printf("\nNEW RECORD: Max fleet size tightened to %d units.\n", MaxRadars)
			saveSystemState()
		}
	}

	// 4. AGGRESSIVE PRUNING WITH NEWBORN PROTECTION
	if successRate >= 95.0 {
		// Sort by kills (Ascending)
		sort.Slice(radarIDs, func(i, j int) bool {
			return kills[radarIDs[i]] < kills[radarIDs[j]]
		})

		numToDrop := int(float64(rCount) * 0.15)
		dropped := 0
		now := simClock.Unix() // Use the simulation clock for timing

		// 6-hour protection window in simulation time
		gracePeriodSeconds := int64(6 * time.Hour.Seconds())

		for _, id := range radarIDs {
			// Stop if we've dropped enough or hit the floor
			if dropped >= numToDrop || (len(radarIDs)-dropped) <= MinRadars {
				break
			}

			e, exists := entities[id]
			if !exists {
				continue
			}

			// --- THE FIX: NEWBORN GRACE PERIOD ---
			// If the radar is younger than 6 hours, skip it.
			// It hasn't had enough "exposure time" to prove its worth.
			if (now - e.StartTime) < gracePeriodSeconds {
				continue
			}

			// If it's old enough and still has 0 or low kills, prune it
			delete(entities, id)
			delete(kills, id)
			dropped++

			// Refund/Efficiency Reward logic
			scalingFactor := 1.0 + (100.0 / float64(len(entities)-len(cityNames)+1))
			budget += EfficiencyBonus * scalingFactor
		}
		saveSystemState()
	}

	// 5. CRISIS RECOVERY
	if budget < BankruptcyLimit || (rCount < MinRadars && budget < RadarCost) {
		fmt.Println("\nCRISIS: Adaptive Mutation Triggered...")
		brain.AdaptiveMutate(successRate)
		budget = EmergencyBudget // Emergency Capital Injection
	}

	lastEraSuccess = successRate

	// 6. FINALIZE ERA & CLOCK SNAP
	for id := range kills {
		kills[id] = 0
	}
	currentCycle++

	// FIX: Explicitly snap eraStartTime to prevent simClock drift
	eraStartTime = eraStartTime.Add(EraDuration)
	simClock = eraStartTime

	wallStart = time.Now()
	totalThreats, totalIntercepts = 0, 0
	for i := range quadrantMisses {
		quadrantMisses[i] = 0
	}
}

// getSatellitePos calculates the current Lat/Lon based on launch origin and orbital mechanics.
func getSatellitePos(startTime int64, launchLat, launchLon float64, currentTime time.Time) (float64, float64) {
	// 1. Calculate time elapsed since launch in hours
	elapsed := currentTime.Sub(time.Unix(startTime, 0)).Hours()

	// 2. LATITUDE (Sine wave with phase shift)
	// We use the launchLat to determine the phase shift so the satellite
	// starts at the launch site, not the equator.
	amplitude := 60.0

	// Ensure launchLat is within the sine amplitude range (-60 to 60)
	clampedLat := math.Max(-amplitude, math.Min(amplitude, launchLat))
	phase := math.Asin(clampedLat / amplitude)

	// Period of 1.5 hours (2 * Pi / 1.5)
	periodFactor := math.Pi / 0.75
	lat := amplitude * math.Sin((elapsed*periodFactor)+phase)

	// 3. LONGITUDE (Linear movement)
	// Longitude increases based on orbital speed.
	lon := launchLon + (elapsed * 25.0)

	// Wrap around the globe correctly to stay within -180 to 180
	lon = math.Mod(lon+180, 360)
	if lon < 0 {
		lon += 360
	}
	lon -= 180

	return lat, lon
}

func getDistanceKM(lat1, lon1, lat2, lon2 float64) float64 {
	p1, p2 := lat1*math.Pi/180, lat2*math.Pi/180
	dp, dl := (lat2-lat1)*math.Pi/180, (lon2-lon1)*math.Pi/180
	a := math.Sin(dp/2)*math.Sin(dp/2) + math.Cos(p1)*math.Cos(p2)*math.Sin(dl/2)*math.Sin(dl/2)
	return EarthRadius * 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
}

func getTetheredCoords() (float64, float64) {
	// Pick a random city to anchor the new radar
	city := entities[cityNames[rand.Intn(len(cityNames))]]

	// Random offset within the tether radius (~7-8 degrees is roughly 800km)
	lat := city.Lat + (rand.Float64()-0.5)*14.0
	lon := city.Lon + (rand.Float64()-0.5)*14.0

	// Safety check: ensure it's actually within range
	if getDistanceKM(city.Lat, city.Lon, lat, lon) > TetherRadiusKM {
		return city.Lat, city.Lon
	}
	return lat, lon
}

func simulateHomingIntercept(sensor *Entity, target *Entity, threatLat, threatLon float64) bool {
	// 1. Base Probability (Space assets start higher but are more sensitive to physics)
	pk := 0.75
	if sensor.Type == "SAT" {
		pk = 0.85

		// 2. TRUE DYNAMIC RELATIVE VELOCITY (Fix)
		// We look 10s into the future to establish a velocity vector
		currLat, currLon := sensor.Lat, sensor.Lon
		futureLat, futureLon := getSatellitePos(sensor.StartTime, sensor.LaunchLat, sensor.LaunchLon, simClock.Add(10*time.Second))

		distNow := getDistanceKM(currLat, currLon, threatLat, threatLon)
		distFuture := getDistanceKM(futureLat, futureLon, threatLat, threatLon)

		// relativeVelocityFactor: Positive means head-on closing, Negative means tail-chase
		// At 25,000 KM/H, this factor is roughly 7.0 KM/s
		relativeVelocityFactor := (distNow - distFuture) / 10.0

		// Apply the Difficulty Multiplier to the threat speed
		// This makes high-speed "Tail-chase" intercepts mathematically near-impossible
		threatDifficulty := 1.0 + (difficultyMultiplier * 0.2) // Scales with your win streak

		if relativeVelocityFactor > 0 {
			// Head-on: Closing speed helps, but difficulty reduces the window
			pk += (0.15 * (relativeVelocityFactor / 7.0)) / threatDifficulty
		} else {
			// Tail-chase: Drastic penalty. If the threat is faster than the SAT's ability
			// to maneuver (relativeVelocityFactor is highly negative), Pk drops to near zero.
			pk += (relativeVelocityFactor / 5.0) * threatDifficulty
		}
	}

	// 3. Distance Decay (Non-Linear)
	// Real interceptors lose energy exponentially at the edge of their envelope.
	distToTarget := getDistanceKM(sensor.Lat, sensor.Lon, target.Lat, target.Lon)
	maxRange := RadarRadiusKM
	if sensor.Type == "SAT" {
		maxRange = SatelliteRangeKM
	}

	// Use a power function to make the "edge" of the radar range much deadlier for accuracy
	distanceFactor := math.Pow(1.0-(distToTarget/maxRange), 1.5)

	// Apply difficulty to the distance factor - making the "effective" range smaller
	finalPk := pk * (0.6 + 0.4*distanceFactor) / difficultyMultiplier

	// 4. Hard Floor/Cap
	if finalPk < 0.05 {
		finalPk = 0.05
	} // Never 0, always a "lucky" chance
	if finalPk > 0.98 {
		finalPk = 0.98
	} // Never 100%, "system failure" chance

	return rand.Float64() < finalPk
}

// getTargetedTether finds a city specifically within the quadrant the AI wants to reinforce.
func getTargetedTether(quadrant int) (float64, float64) {
	var candidates []string

	mu.RLock()
	for _, name := range cityNames {
		city := entities[name]
		// Map Lat/Lon to 0-3 Quadrant ID (NW: 0, NE: 1, SW: 2, SE: 3)
		qIdx := 0
		if city.Lat < 0 {
			qIdx += 2
		} // South
		if city.Lon > 0 {
			qIdx += 1
		} // East

		if qIdx == quadrant {
			candidates = append(candidates, name)
		}
	}
	mu.RUnlock()

	// Fallback: If no cities exist in that specific quadrant, use global random tether
	if len(candidates) == 0 {
		return getTetheredCoords()
	}

	// Pick a random city from the correct quadrant
	targetCityName := candidates[rand.Intn(len(candidates))]
	mu.RLock()
	targetCity := entities[targetCityName]
	lat, lon := targetCity.Lat, targetCity.Lon
	mu.RUnlock()

	// Apply tethering radius (800km) to scatter radars around the city
	lat += (rand.Float64() - 0.5) * 14.0
	lon += (rand.Float64() - 0.5) * 14.0

	return lat, lon
}

func runPhysicsEngine() {
	var lastLaunch time.Time
	mu.Lock()
	if simClock.IsZero() {
		simClock = time.Now()
		eraStartTime = simClock
	}
	mu.Unlock()

	for {
		mu.Lock()
		if simClock.After(eraStartTime.Add(EraDuration)) || forceReset {
			forceReset = false
			mu.Unlock()
			autonomousEraReset()
			continue
		}

		// Update Satellites
		satCount := 0
		for _, e := range entities {
			if e.Type == "SAT" {
				satCount++
			}
		}
		if satCount < MaxSatellites && simClock.Sub(lastLaunch) >= LaunchInterval {
			site := launchSites[rand.Intn(len(launchSites))]
			id := fmt.Sprintf("SAT-%d-%d", currentCycle, satCount)
			entities[id] = &Entity{ID: id, Type: "SAT", LaunchLat: site.Lat, LaunchLon: site.Lon, Lat: site.Lat, Lon: site.Lon, StartTime: simClock.Unix()}
			lastLaunch = simClock
		}
		for _, e := range entities {
			if e.Type == "SAT" {
				e.Lat, e.Lon = getSatellitePos(e.StartTime, e.LaunchLat, e.LaunchLon, simClock)
			}
		}

		// Snapshot for Lock-Free Simulation
		type EntityVal struct {
			ID, Type string
			Lat, Lon float64
		}
		snap := make(map[string]EntityVal)
		spatialIndex := make(map[int][]string)
		cityIndex := make(map[int][]string)
		quadrantRadars := make([]float64, 4)
		rCount := 0

		for id, e := range entities {
			snap[id] = EntityVal{ID: e.ID, Type: e.Type, Lat: e.Lat, Lon: e.Lon}
			gID := getGridID(e.Lat, e.Lon)
			if e.Type == "RADAR" {
				spatialIndex[gID] = append(spatialIndex[gID], id)
				rCount++
				idx := 0
				if e.Lat < 0 {
					idx += 2
				}
				if e.Lon > 0 {
					idx += 1
				}
				quadrantRadars[idx]++
			} else if e.Type == "SAT" {
				spatialIndex[gID] = append(spatialIndex[gID], id)
			} else if e.Type == "CITY" {
				cityIndex[gID] = append(cityIndex[gID], id)
			}
		}
		snapBudget, snapSuccess := budget, successRate
		mu.Unlock()

		// Worker Pool
		var wg sync.WaitGroup
		numWorkers := runtime.NumCPU()
		batchSize := 200
		workerKills := make(chan map[string]int, numWorkers)
		workerStats := make(chan struct {
			ints, thrs int
			rew, pen   float64
			misses     []float64
		}, numWorkers)

		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				lKills := make(map[string]int)
				lMisses := make([]float64, 4)
				li, lt, lr, lp := 0, 0, 0.0, 0.0
				for i := 0; i < batchSize/numWorkers; i++ {
					sLat, sLon := (rand.Float64()*180.0)-90.0, (rand.Float64()*360.0)-180.0
					gridID := getGridID(sLat, sLon)
					var targetCity EntityVal
					foundCity := false
				citySearch:
					for r := -3; r <= 3; r++ {
						for c := -3; c <= 3; c++ {
							if names, ok := cityIndex[gridID+(r*Cols)+c]; ok {
								for _, n := range names {
									if getDistanceKM(sLat, sLon, snap[n].Lat, snap[n].Lon) < MissileMaxRangeKM {
										targetCity, foundCity = snap[n], true
										break citySearch
									}
								}
							}
						}
					}
					if !foundCity {
						continue
					}
					intercepted := false
					tGrid := getGridID(targetCity.Lat, targetCity.Lon)
				outer:
					for r := -2; r <= 2; r++ {
						for c := -2; c <= 2; c++ {
							if sids, ok := spatialIndex[tGrid+(r*Cols)+c]; ok {
								for _, sid := range sids {
									sensor := snap[sid]
									rng := RadarRadiusKM
									if sensor.Type == "SAT" {
										rng = SatelliteRangeKM
									}
									if getDistanceKM(sensor.Lat, sensor.Lon, targetCity.Lat, targetCity.Lon) < rng {
										if simulateHomingIntercept(&Entity{ID: sensor.ID, Type: sensor.Type, Lat: sensor.Lat, Lon: sensor.Lon}, &Entity{Lat: targetCity.Lat, Lon: targetCity.Lon}, sLat, sLon) {
											lKills[sid]++
											li++
											lr += InterceptReward
											intercepted = true
											break outer
										}
									}
								}
							}
						}
					}
					if !intercepted {
						lt++
						// FIX: DYNAMIC PENALTY (Prevents the "Crisis Loop")
						p := CityImpactPenalty
						if snapSuccess < 85.0 {
							p *= 0.05
						} // 95% discount while learning
						lp += p
						idx := 0
						if targetCity.Lat < 0 {
							idx += 2
						}
						if targetCity.Lon > 0 {
							idx += 1
						}
						lMisses[idx]++
					}
				}
				workerKills <- lKills
				workerStats <- struct {
					ints, thrs int
					rew, pen   float64
					misses     []float64
				}{li, lt, lr, lp, lMisses}
			}()
		}

		go func() { wg.Wait(); close(workerKills); close(workerStats) }()

		mu.Lock()
		for lk := range workerKills {
			for id, count := range lk {
				if _, ok := kills[id]; ok {
					kills[id] += count
				}
			}
		}
		for ls := range workerStats {
			totalIntercepts += ls.ints
			totalThreats += ls.thrs
			budget += (ls.rew - ls.pen)
			for i, v := range ls.misses {
				quadrantMisses[i] += v
			}
		}

		simClock = simClock.Add(TimeStepping)
		if totalThreats+totalIntercepts > 0 {
			successRate = (float64(totalIntercepts) / float64(totalThreats+totalIntercepts)) * 100
		}

		// NN Prediction & Actions
		dayProgress := simClock.Sub(eraStartTime).Seconds() / EraDuration.Seconds()
		input := []float64{
			math.Max(-1, math.Min(snapBudget/1e9, 1.0)), float64(rCount) / float64(MaxRadars),
			snapSuccess / 100.0, math.Min(quadrantMisses[0]/500.0, 1.0), math.Min(quadrantMisses[1]/500.0, 1.0),
			math.Min(quadrantMisses[2]/500.0, 1.0), math.Min(quadrantMisses[3]/500.0, 1.0),
			float64(satCount) / float64(MaxSatellites), math.Min(quadrantRadars[0]/20.0, 1.0),
			math.Min(quadrantRadars[1]/20.0, 1.0), math.Min(quadrantRadars[2]/20.0, 1.0),
			math.Min(quadrantRadars[3]/20.0, 1.0), (successRate - lastEraSuccess) / 100.0,
			difficultyMultiplier / 10.0, math.Sin(2 * math.Pi * dayProgress), math.Cos(2 * math.Pi * dayProgress),
		}

		action, latN, lonN := brain.PredictSpatial(input)
		prec := math.Max(5.0, 20.0*(1.0-(successRate/100.0)))
		if action == 1 && rCount < MaxRadars {
			tQ := 0
			maxM := -1.0
			for i, m := range quadrantMisses {
				if m > maxM {
					maxM = m
					tQ = i
				}
			}
			bLat, bLon := getTargetedTether(tQ)
			id := fmt.Sprintf("R-AI-%d-%d", currentCycle, rand.Intn(1e7))
			entities[id] = &Entity{ID: id, Type: "RADAR", Lat: bLat + (latN * prec), Lon: bLon + (lonN * prec), StartTime: simClock.Unix()}
			kills[id] = 0
			if budget >= RadarCost {
				budget -= RadarCost
			}
		} else if action == 2 && budget >= RelocationCost {
			var wID string
			minK := 999999
			for id, c := range kills {
				if e, ok := entities[id]; ok && e.Type == "RADAR" && c < minK {
					minK = c
					wID = id
				}
			}
			if e, ok := entities[wID]; ok {
				bL, bO := getTetheredCoords()
				e.Lat, e.Lon = bL+(latN*prec), bO+(lonN*prec)
				e.LastMoved = time.Now().UnixMilli()
				kills[wID] = 0
				budget -= RelocationCost
			}
		}
		mu.Unlock()
		runtime.Gosched()
		time.Sleep(1 * time.Millisecond)
	}
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())

	// 1. POPULATE CITIES
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

	// 2. INITIALIZE NEURAL NETWORK
	// Use a small multiplier for initial weight scaling to prevent saturation
	brain = NewBrain(mutationRate)

	// 3. LOAD PERSISTENCE (Crucial for NN Persistence)
	// We call this before seeding so we know if we actually need a new fleet
	loadSystemState()

	// 4. SEED FLEET (Only if no optimized fleet was loaded)
	mu.Lock()
	radarCount := 0
	for _, e := range entities {
		if e.Type == "RADAR" {
			radarCount++
		}
	}
	if radarCount < MinRadars {
		fmt.Println("NO PERSISTENCE FOUND. INITIALIZING SEED FLEET...")
		for i := 0; i < MinRadars; i++ {
			id := fmt.Sprintf("R-START-%d", i)
			lat, lon := getTetheredCoords()
			entities[id] = &Entity{ID: id, Type: "RADAR", Lat: lat, Lon: lon, StartTime: time.Now().Unix()}
		}
	} else {
		fmt.Printf("RESUMING ERA %d WITH %d OPTIMIZED ASSETS\n", currentCycle, radarCount)
	}
	mu.Unlock()

	wallStart = time.Now()
	go runPhysicsEngine()

	// 5. HTTP HANDLERS
	http.HandleFunc("/panic", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		entities = make(map[string]*Entity)
		cityNames = []string{}
		kills = make(map[string]int)
		for name, pos := range cityData {
			entities[name] = &Entity{ID: name, Type: "CITY", Lat: pos[0], Lon: pos[1]}
			cityNames = append(cityNames, name)
		}
		for i := 0; i < MinRadars; i++ {
			id := fmt.Sprintf("R-START-%d", i)
			lat, lon := getTetheredCoords()
			entities[id] = &Entity{ID: id, Type: "RADAR", Lat: lat, Lon: lon, StartTime: time.Now().Unix()}
		}
		forceReset = true
		w.Write([]byte("Shield Re-initialized Successfully"))
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

		// UPDATED: Added max_radars for the UI dashboard
		json.NewEncoder(w).Encode(map[string]interface{}{
			"cycle":         currentCycle,
			"budget":        budget,
			"entities":      all,
			"success":       successRate,
			"streak":        winStreakCounter,
			"isOver":        isSimulationOver,
			"yps":           yps,
			"max_radars":    MaxRadars,
			"mutation_rate": mutationRate,
		})
	})

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		template.Must(template.New("v").Parse(uiHTML)).Execute(w, nil)
	})

	fmt.Println("AEGIS RUNNING AT :8080")
	http.ListenAndServe(":8080", nil)
}

const uiHTML = `
<!DOCTYPE html><html><head><title>AEGIS REAL-TIME MONITOR</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
    body { margin:0; background:#001; color:#0f0; font-family:'Courier New', monospace; overflow:hidden; } 
    #stats { 
        position:fixed; top:15px; left:15px; z-index:1000; 
        background:rgba(0,15,30,0.95); padding:20px; border:2px solid #0af; 
        box-shadow: 0 0 20px #0af; border-radius: 4px; width: 280px;
    } 
    #map { height:100vh; width:100vw; background: #000; }
    .stat-line { font-size: 1.1em; margin-bottom: 5px; border-bottom: 1px solid #0af3; }
    .val { float: right; color: #fff; padding-left: 20px; }
    .highlight { color: #f0f; }
    .intel-box { 
        margin-top: 10px; padding: 10px; border: 1px solid #ff0; 
        background: rgba(255, 255, 0, 0.05); text-align: center;
    }
    #intensity { font-weight: bold; font-size: 1.2em; display: block; }
    #lr-val { font-size: 0.8em; color: #aaa; }
    #panic-btn {
        width: 100%; margin-top: 15px; padding: 10px;
        background: #400; color: #f00; border: 1px solid #f00;
        cursor: pointer; font-family: inherit; font-weight: bold;
        transition: 0.3s;
    }
    #panic-btn:hover { background: #f00; color: #000; }
</style></head>
<body>
<div id="stats">
    <div class="stat-line">SYSTEM STATUS: <span id="status" class="val" style="color:#0f0">ACTIVE</span></div>
    <div class="stat-line">ERA: <span id="era" class="val">0</span></div>
    <div class="stat-line">FLEET: <span id="rcount" class="val">0</span></div>
    <div class="stat-line">EFFICIENCY: <span id="success" class="val">0</span>%</div>
    <div class="stat-line">THROUGHPUT: <span id="yps" class="val">0</span> Y/sec</div>
    <div class="stat-line">BUDGET: <span id="budget" class="val" style="color:#fb0">$0</span></div>
    
    <div class="intel-box">
        <span id="intensity">ANALYZING...</span>
        <span id="lr-val">LR: 0.00000</span>
    </div>

    <button id="panic-btn" onclick="triggerPanic()">MANUAL SYSTEM RESET</button>
</div>
<div id="map"></div>
<script>
    var map = L.map('map', { zoomControl:false, attributionControl:false, preferCanvas: true }).setView([25, 10], 2);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png').addTo(map);
    
    var layers = {};
    var isFetching = false;

    async function triggerPanic() {
        document.getElementById('status').innerText = "REBOOTING...";
        document.getElementById('status').style.color = "#f00";
        try { await fetch('/panic'); } catch (e) { console.error("Reset failed:", e); }
    }

    async function updateUI() {
        if (isFetching) return;
        isFetching = true;

        try {
            const response = await fetch('/intel');
            const data = await response.json();

            // 1. Sync Standard Stats
            document.getElementById('era').innerText = data.cycle;
            document.getElementById('success').innerText = (data.success || 0).toFixed(2);
            document.getElementById('budget').innerText = "$" + Math.floor(data.budget).toLocaleString();
            document.getElementById('yps').innerText = (data.yps || 0).toFixed(4);
            
            // 2. Intelligence & Training UI Logic
            const yps = data.yps || 0;
            const success = data.success || 0;
            const lr = data.mutation_rate || 0;
            const intensityEl = document.getElementById('intensity');
            
            document.getElementById('lr-val').innerText = "MUTATION RATE: " + lr.toFixed(5);

            if (success < 75) {
                intensityEl.innerText = yps > 5 ? "HYPER-LEARNING" : "CRISIS RECOVERY";
                intensityEl.style.color = "#f44";
            } else if (success < 95) {
                intensityEl.innerText = "OPTIMIZING";
                intensityEl.style.color = "#fb0";
            } else {
                intensityEl.innerText = "FINE-TUNING";
                intensityEl.style.color = "#0af";
            }

            // 3. Entity Rendering
            const now = Date.now();
            const currentIds = new Set();
            let radarCount = 0;

            (data.entities || []).forEach(e => {
                currentIds.add(e.id);
                if (e.type === 'RADAR') radarCount++;

                if (!layers[e.id]) {
                    if (e.type === 'RADAR') {
                        layers[e.id] = L.circle([e.lat, e.lon], { radius: 1200000, color: '#0f0', weight: 1, fillOpacity: 0.1 }).addTo(map);
                    } else if (e.type === 'SAT') {
                        layers[e.id] = L.circle([e.lat, e.lon], { radius: 2500000, color: '#0af', weight: 1, fillOpacity: 0.05, dashArray: '5, 10' }).addTo(map);
                    } else {
                        layers[e.id] = L.circleMarker([e.lat, e.lon], { radius: 3, color: '#f44', fillOpacity: 0.7 }).addTo(map);
                    }
                } else {
                    layers[e.id].setLatLng([e.lat, e.lon]);
                    const movedRecently = e.last_moved && (now - e.last_moved < 1500);
                    if (movedRecently) {
                        layers[e.id].setStyle({ color: '#ff0', weight: 6, fillOpacity: 0.4 });
                    } else {
                        if (e.type === 'RADAR') layers[e.id].setStyle({color: '#0f0', weight: 1, fillOpacity: 0.1});
                        else if (e.type === 'SAT') layers[e.id].setStyle({color: '#0af', weight: 1, fillOpacity: 0.05});
                    }
                }
            });

            document.getElementById('rcount').innerText = radarCount;

            for (let id in layers) {
                if (!currentIds.has(id)) { map.removeLayer(layers[id]); delete layers[id]; }
            }
        } catch (e) { console.error("UI Sync drop:", e); }
        isFetching = false;
    }
    
    setInterval(updateUI, 150);
</script></body></html>`

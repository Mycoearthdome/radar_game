package main

import (
	"bytes"
	"context"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net"
	"os"
	"os/signal"
	"runtime"
	"sort"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"github.com/InternatBlackhole/cudago/cuda"

	_ "embed"
)

//go:embed radar.ptx
var ptxFile []byte

const (
	RadarRadiusKM           = 1200.0
	TetherRadiusKM          = 800.0
	RadarCost               = 250000.0
	RelocationCost          = 5000.0
	EnforcementFee          = 5000000.0
	MinRadars               = 120
	CityImpactPenalty       = 200000.0
	InterceptReward         = 10000.0
	EarthRadius             = 6371.0
	EraDuration             = 24 * time.Hour
	TimeStepping            = 10 * time.Second
	RadarFile               = "RADAR.json"
	BrainFile               = "BRAIN.gob"
	BankruptcyLimit         = 0.0
	GridSize                = 10 // 10 degree cells
	Cols                    = 36 // 360 / 10
	Rows                    = 18 // 180 / 10
	TargetSuccess           = 100.0
	RequiredWinStreak       = 10     // Number of eras to maintain 100% before "winning"
	MissileMaxSpeedKMH      = 8000.0 // Hypersonic Mach 6.5+
	MissileGMLimit          = 40.0   // Max turning force
	ProximityRadiusKM       = 5.0    // Blast zone
	KineticRadiusKM         = 0.1    // Direct hit
	BaseKillProbability     = 0.85   // Success rate for proximity detonation
	MaxEnergy               = 100.0  // Starting energy percentage
	DragPenaltyBase         = 0.05   // Energy loss per KM of flight
	ManeuverEnergyCost      = 2.5    // Extra cost for high-G turns
	MaxSatellites           = 20
	SatelliteRangeKM        = 2500.0                  // Higher vantage point = wider reach
	LaunchInterval          = 1 * 24 * 31 * time.Hour //one month
	MissileMaxRangeKM       = 2500.0                  // The absolute maximum distance a missile can travel
	FuelConsumption         = 0.04                    // Energy/Fuel cost per KM traveled by the threat
	StartBudget             = 1000000000.0
	EmergencyBudget         = StartBudget
	MissilesBatchedInFlight = 1000
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
	mutationRate         = 0.1                //was 1.0
	quadrantMisses       = make([]float64, 4) // [NW, NE, SW, SE]
	winStreakCounter     = 0
	isSimulationOver     = false
	EfficiencyBonus      = 100000.0
	difficultyMultiplier = 1.0
	ctx, cancel          = context.WithCancel(context.Background())
	activeRadars         []*Entity
	batchInputs          [][]float64
	gpuJobs              chan Job
)

var launchSites = []struct{ Lat, Lon float64 }{
	{28.57, -80.64},  // Cape Canaveral
	{34.74, -120.57}, // Vandenberg
	{5.23, -52.76},   // Kourou
	{45.96, 63.30},   // Baikonur
	{18.65, 100.48},  // Jiuquan
}

type Job struct {
	Run func()
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
	mu sync.Mutex

	// GPU Memory objects
	// These hold the metadata for the allocations on your RTX 3060
	w0_gpu *cuda.DeviceMemory
	w1_gpu *cuda.DeviceMemory

	// Local CPU copies for persistence and mutation
	// Used for saving to BRAIN.gob and performing genetic crossovers
	w0_cpu []float64
	w1_cpu []float64

	// The compiled CUDA kernel loaded from your .cubin or .ptx
	kernel *cuda.Function
}

func (b *Brain) SyncToGPU(context *cuda.PrimaryCtx) { // Pass the context in
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Force this thread to use our primary context
	cuda.SetCurrentContext(&context.Context)
	// 1. SAFETY CHECK: Ensure CPU slices are not empty or nil
	if len(b.w0_cpu) == 0 || len(b.w1_cpu) == 0 {
		log.Println("WARNING: SyncToGPU skipped - CPU weights are empty.")
		return
	}

	// 2. SAFETY CHECK: Ensure GPU memory was actually allocated
	if b.w0_gpu == nil || b.w1_gpu == nil {
		log.Println("ERROR: SyncToGPU failed - GPU device memory not initialized.")
		return
	}

	size0 := uint64(len(b.w0_cpu) * 8)
	size1 := uint64(len(b.w1_cpu) * 8)

	// 3. Perform the Memcpy
	res0 := b.w0_gpu.MemcpyToDevice(unsafe.Pointer(&b.w0_cpu[0]), size0)
	if res0 != nil {
		log.Printf("Error syncing w0: %s", res0.Error())
		panic(res0.Error())
	}

	res1 := b.w1_gpu.MemcpyToDevice(unsafe.Pointer(&b.w1_cpu[0]), size1)
	if res1 != nil {
		log.Printf("Error syncing w1: %s", res1.Error())
		panic(res1.Error())
	}

	log.Println("NN weights successfully UPLOADED to GPU.")
}

func (b *Brain) SyncFromGpu() {

	var p_w0 runtime.Pinner
	// Pin both slices so the GC doesn't move them during the transfer
	p_w0.Pin(&b.w0_cpu[0])
	defer p_w0.Unpin()

	var p_w1 runtime.Pinner
	p_w1.Pin(&b.w1_cpu[0])
	defer p_w1.Unpin()

	// Size in bytes: number of elements * 8 (float64)
	size0 := uint64(len(b.w0_cpu) * 8)
	size1 := uint64(len(b.w1_cpu) * 8)

	// Using MemcpyFromDevice(dest unsafe.Pointer, srcSize uint64)
	// This pulls the data from GPU memory (w0_gpu) into the CPU slice (w0_cpu)
	res0 := b.w0_gpu.MemcpyFromDevice(unsafe.Pointer(&b.w0_cpu[0]), size0)
	if res0 != nil {
		log.Printf("Error downloading w0: %s", res0.Error())
		panic(res0.Error())
	}

	res1 := b.w1_gpu.MemcpyFromDevice(unsafe.Pointer(&b.w1_cpu[0]), size1)
	if res1 != nil {
		log.Printf("Error downloading w1: %s", res1.Error())
		panic(res1.Error())
	}

	log.Println("NN weights successfully DOWNLOADED from GPU.")
}

func (b *Brain) SyncDataToGPU(d_in *cuda.DeviceMemory, flatInputs []float64) {
	size := uint64(len(flatInputs) * 8)

	result := d_in.MemcpyToDevice(unsafe.Pointer(&flatInputs[0]), size)
	if result != nil {
		log.Printf("Error UPLOADING DATA to GPU function: %s", result.Error())
		panic(result.Error())
	}
}

func (b *Brain) SyncDataFromGPU(dest unsafe.Pointer, d_out *cuda.DeviceMemory, size uint64) {
	// Size is passed as the number of bytes (elements * 8)
	result := d_out.MemcpyFromDevice(dest, size)
	if result != nil {
		log.Printf("Error DOWNLOADING DATA to GPU function: %s", result.Error())
		panic(result.Error())
	}
}

func (b *Brain) Evaluate(input []float64) []float64 {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Wrap single input in a batch
	batch := [][]float64{input}

	// Call GPU batch prediction
	outputs, err := b.PredictBatch(batch)
	if err != nil {
		log.Printf("[Brain] Evaluate error: %v", err)
		return nil
	}

	// Return the first (and only) output
	if len(outputs) == 0 {
		return nil
	}

	return outputs[0]
}

func (b *Brain) Mutate(rate float64, context *cuda.PrimaryCtx) {

	for i := range b.w0_cpu {
		if rand.Float64() < 0.9 {
			b.w0_cpu[i] += rand.NormFloat64() * rate
		} else {
			b.w0_cpu[i] *= (1.0 + rand.NormFloat64()*rate)
		}
	}

	for i := range b.w1_cpu {
		if rand.Float64() < 0.9 {
			b.w1_cpu[i] += rand.NormFloat64() * rate
		} else {
			b.w1_cpu[i] *= (1.0 + rand.NormFloat64()*rate)
		}
	}

	// Correct GPU sync
	b.SyncToGPU(context)
}

// Helper to get grid ID from coordinates
func getGridID(lat, lon float64) int {
	row := int((lat + 90) / GridSize)
	col := int((lon + 180) / GridSize)
	return row*Cols + col
}

func NewCudaBrain() *Brain {
	// 1. Setup dimensions to match your game.go logic
	const (
		inputSize  = 18
		hiddenSize = 24
		outputSize = 3
	)

	// 1. Load the PTX Module compiled by your compile.sh
	// "radar.ptx" is the output from: nvcc -ptx radar.cu -o radar.ptx
	mod, err := cuda.LoadModuleData(ptxFile)
	if err != nil {
		log.Fatalf("Failed to load PTX module: %v", err)
	}

	// 2. Get the function handle for your predictBatch kernel
	kernel, err := mod.GetFunction("predictBatch")
	if err != nil {
		log.Fatalf("Failed to get kernel function: %v", err)
	}

	// 2. Allocate CPU memory
	w0 := make([]float64, inputSize*hiddenSize)
	w1 := make([]float64, hiddenSize*outputSize)

	// 3. Manually apply Glorot/Xavier Initialization
	// Formula: random value between ±sqrt(6 / (inputs + outputs))
	limit0 := math.Sqrt(6.0 / float64(inputSize+hiddenSize))
	for i := range w0 {
		w0[i] = (rand.Float64()*2 - 1) * limit0
	}

	limit1 := math.Sqrt(6.0 / float64(hiddenSize+outputSize))
	for i := range w1 {
		w1[i] = (rand.Float64()*2 - 1) * limit1
	}

	// 4. Allocate GPU memory via CudaGo
	w0_gpu, result := cuda.DeviceMemAlloc(uint64(len(w0) * 8))
	if result != nil {
		log.Printf("Error allocating w0: %v", result.Code())
		panic(result.Error())
	}

	w1_gpu, result := cuda.DeviceMemAlloc(uint64(len(w1) * 8))
	if result != nil {
		log.Printf("Error allocating w1: %v", result.Code())
		panic(result.Error())
	}

	// 5. Initial Sync
	// Size in bytes: number of float64 elements * 8
	size0 := uint64(len(w0) * 8)
	size1 := uint64(len(w1) * 8)

	// Using the method: MemcpyToDevice(src unsafe.Pointer, srcSize uint64)
	// We get the pointer to the first element of the CPU slice
	res0 := w0_gpu.MemcpyToDevice(unsafe.Pointer(&w0[0]), size0)
	if res0 != nil {
		log.Printf("Error syncing w0: %v", res0)
	}

	res1 := w1_gpu.MemcpyToDevice(unsafe.Pointer(&w1[0]), size1)
	if res1 != nil {
		log.Printf("Error syncing w1: %v", res1)
	}

	log.Println("NN weights successfully UPLOADED to GPU.")

	return &Brain{
		w0_cpu: w0,
		w1_cpu: w1,
		w0_gpu: w0_gpu,
		w1_gpu: w1_gpu,
		kernel: kernel,
	}
}

func (b *Brain) AdaptiveMutate(success float64, context *cuda.PrimaryCtx) {
	// 1. Calculate Dynamic Jitter Rate
	rate := 0.2 * (1.0 - (success / 100.0))
	if rate < 0.002 {
		rate = 0.002
	}

	mutationRate = rate

	mutateSlice := func(s []float64) {
		for i := range s {
			if rand.Float64() > (success / 100.0) {
				s[i] += rand.NormFloat64() * rate
			}
		}
	}

	b.mu.Lock()
	mutateSlice(b.w0_cpu)
	mutateSlice(b.w1_cpu)
	b.mu.Unlock()

	// Synchronize to GPU
	b.SyncToGPU(context) // assumes SyncToGPU handles setting CUDA context internally

	log.Printf("[Brain] Mutated weights with rate %.5f based on %.2f%% success\n", rate, success)
}

func (b *Brain) Train(inputs []float64, idealAction []float64, context *cuda.PrimaryCtx) {
	// We are already inside the GPU Worker goroutine here.
	// The CUDA Context is set, and we are not holding the global 'mu' lock.

	// 1. Calculate Error / Backprop Trigger
	// In your genetic algorithm, we use the global success rate to determine
	// if we should scramble weights (AdaptiveMutate) or refine them (Mutate).

	if successRate < 95.0 {
		// Low success: Needs major structural changes
		// We call AdaptiveMutate directly. It handles its own locking internally.
		b.AdaptiveMutate(successRate, context)
	} else {
		// High success: Fine-tuning only
		// We use a small mutation rate to settle into the local minimum
		b.Mutate(0.002, context)
	}

	// Note: We do not need to calculate gradients here because this is a
	// genetic/evolutionary algorithm, not standard backpropagation.
}

func (b *Brain) PredictBatch(batchInputs [][]float64) ([][]float64, error) {

	numRadars := len(batchInputs)
	if numRadars == 0 {
		return nil, nil
	}

	// 1. Flatten the 2D slice into a 1D slice for the GPU
	// Using a pre-allocated flat slice reduces GC pressure during the physics loop.
	flatInputs := make([]float64, 0, numRadars*18)
	for _, radar := range batchInputs {
		flatInputs = append(flatInputs, radar...)
	}

	if len(flatInputs) == 0 {
		return nil, errors.New("empty input flatten result")
	}

	// 2. PIN THE MEMORY: This tells the Go GC not to move flatInputs
	var pinIn runtime.Pinner
	pinIn.Pin(&flatInputs[0])
	defer pinIn.Unpin()

	// 3. Allocate temporary GPU memory
	// DeviceMemAlloc is the standard for temporary buffers in this library.
	d_in, err := cuda.DeviceMemAlloc(uint64(len(flatInputs) * 8))
	if err != nil {
		return nil, fmt.Errorf("failed to allocate input buffer: %s", err.Error())
	}
	defer d_in.Free() // Ensure cleanup even on error

	d_out, err := cuda.DeviceMemAlloc(uint64(numRadars * 3 * 8))
	if err != nil {
		return nil, fmt.Errorf("failed to allocate output buffer: %v", err)
	}
	defer d_out.Free()

	// 4. Sync inputs to the RTX 3060
	b.SyncDataToGPU(d_in, flatInputs)

	// 5. Configure Thread/Block counts
	threadsPerBlock := 256
	blocksPerGrid := (numRadars + threadsPerBlock - 1) / threadsPerBlock

	// 6. Wrap ints into cuda.Dim3 structs
	gridDim := cuda.Dim3{X: uint32(blocksPerGrid), Y: 1, Z: 1}
	blockDim := cuda.Dim3{X: uint32(threadsPerBlock), Y: 1, Z: 1}

	// 7. LAUNCH KERNEL with correct types
	// Device pointers must be local variables
	inPtr := d_in.Ptr
	w0Ptr := b.w0_gpu.Ptr
	w1Ptr := b.w1_gpu.Ptr
	outPtr := d_out.Ptr

	// CUDA kernel expects 32-bit int
	nr := int32(numRadars)

	// ---- Launch kernel ----
	err = b.kernel.Launch(
		gridDim,
		blockDim,
		unsafe.Pointer(&inPtr),
		unsafe.Pointer(&w0Ptr),
		unsafe.Pointer(&w1Ptr),
		unsafe.Pointer(&outPtr),
		unsafe.Pointer(&nr),
	)

	if err != nil {
		return nil, fmt.Errorf("kernel launch failed: %s", err.Error())
	}

	if err := cuda.CurrentContextSynchronize(); err != nil {
		return nil, err
	}

	// 8. Download results to Host (CPU)
	flatOutputs := make([]float64, numRadars*3)
	// 9. PIN THE MEMORY: This tells the Go GC not to move flatOutputs
	var pinOut runtime.Pinner
	pinOut.Pin(&flatOutputs[0])
	defer pinOut.Unpin()

	b.SyncDataFromGPU(unsafe.Pointer(&flatOutputs[0]), d_out, uint64(numRadars*3))

	// 10. Reshape for the Physics Engine
	reshaped := make([][]float64, numRadars)
	for i := 0; i < numRadars; i++ {
		reshaped[i] = flatOutputs[i*3 : (i+1)*3]
	}

	return reshaped, nil
}

// getBrainInputs generates the 16-variable state representation for the NN.
// It matches the input structure used in runPhysicsEngine.
func getBrainInputs(e *Entity, rCount int, snapBudget float64, snapSuccess float64, quadrantRadars []float64, satCount int, dayProgress float64) []float64 {
	// Normalization Helpers
	normBudget := math.Max(-1, math.Min(snapBudget/1e9, 1.0))
	normSuccess := snapSuccess / 100.0
	normDelta := (successRate - lastEraSuccess) / 100.0

	// NEW: Personal Performance Metric
	// We look up this specific entity's kills.
	// Normalized by 10 (assuming a 'good' radar gets 10+ kills per era)
	personalKills := float64(kills[e.ID])
	normKills := math.Min(personalKills/10.0, 1.0)

	// Time context (Cyclical)
	sinTime := math.Sin(2 * math.Pi * dayProgress)
	cosTime := math.Cos(2 * math.Pi * dayProgress)

	return []float64{
		e.Lat / 90.0,                         // 0
		e.Lon / 180.0,                        // 1
		normBudget,                           // 2
		float64(rCount) / float64(MaxRadars), // 3
		normSuccess,                          // 4

		// QUADRANT MISSES
		math.Min(quadrantMisses[0]/100.0, 1.0), // 5
		math.Min(quadrantMisses[1]/100.0, 1.0), // 6
		math.Min(quadrantMisses[2]/100.0, 1.0), // 7
		math.Min(quadrantMisses[3]/100.0, 1.0), // 8

		float64(satCount) / float64(MaxSatellites), // 9

		// QUADRANT RADAR DENSITY
		math.Min(quadrantRadars[0]/20.0, 1.0), // 10
		math.Min(quadrantRadars[1]/20.0, 1.0), // 11
		math.Min(quadrantRadars[2]/20.0, 1.0), // 12
		math.Min(quadrantRadars[3]/20.0, 1.0), // 13

		normDelta, // 14
		normKills, // 15
		sinTime,   // 16
		cosTime,   // 17
	}
}

func loadSystemState(context *cuda.PrimaryCtx) {
	f, err := os.Open(BrainFile)
	if err != nil {
		log.Printf("No persistence found: %v", err)
		return
	}
	defer f.Close()

	// 1. Decode into a temporary "Ghost" brain
	var loadedBrain Brain
	decoder := gob.NewDecoder(f)
	if err := decoder.Decode(&loadedBrain); err != nil {
		log.Printf("Decode error: %v", err)
		return
	}

	// 2. Safely copy only the CPU weights into our active 'brain'
	// This preserves the brain.w0_gpu and brain.w1_gpu pointers we made in main()
	mu.Lock()
	if len(loadedBrain.w0_cpu) > 0 {
		copy(brain.w0_cpu, loadedBrain.w0_cpu)
		copy(brain.w1_cpu, loadedBrain.w1_cpu)
	}
	mu.Unlock()

	// 3. Now it is safe to sync
	brain.SyncToGPU(context)
	log.Println("Persistence loaded and synced to GPU.")
}

func saveSystemState() {
	mu.RLock()
	defer mu.RUnlock()

	// 1. Save NN Weights, Min-Ever Record, and Scaled Limit
	f, err := os.Create(BrainFile)
	if err == nil {
		enc := gob.NewEncoder(f)

		// CRITICAL CHANGE: Encode the raw float64 slices, not Gorgonia Values.
		// These slices are the "Source of Truth" for your NN persistence.
		enc.Encode(brain.w0_cpu)
		enc.Encode(brain.w1_cpu)

		enc.Encode(minEverRadars)
		enc.Encode(MaxRadars)
		f.Close()
	}

	// 2. Save Optimized Entity Positions (Remains the same)
	var optimizedFleet []Entity
	for _, e := range entities {
		if e.Type == "CITY" || e.Type == "RADAR" || e.Type == "SAT" {
			optimizedFleet = append(optimizedFleet, *e)
		}
	}

	jsonData, err := json.MarshalIndent(optimizedFleet, "", "  ")
	if err == nil {
		os.WriteFile(RadarFile, jsonData, 0644)
	}
}

func autonomousEraReset(snapBudget float64, snapSuccess float64, snapDifficulty float64, context *cuda.PrimaryCtx, jobs chan<- Job) {
	// --- 1. Aggregate entities ---
	radarIDs := []string{}
	quadrantRadars := make([]float64, 4)
	satCount := 0

	mu.RLock()
	for id, e := range entities {
		switch e.Type {
		case "RADAR":
			radarIDs = append(radarIDs, id)
			idx := 0
			if e.Lat < 0 {
				idx += 2
			}
			if e.Lon > 0 {
				idx += 1
			}
			quadrantRadars[idx]++
		case "SAT":
			satCount++
		}
	}
	mu.RUnlock()

	rCount := len(radarIDs)
	dayProgress := 1.0

	// --- 2. Train the Brain using Jobs (WITH SYNCHRONIZATION) ---
	// We use a WaitGroup to ensure all training is finished before resetting the era.
	var wg sync.WaitGroup
	for _, id := range radarIDs {
		mu.RLock()
		e, exists := entities[id]
		k := kills[id]
		mu.RUnlock()

		if !exists {
			continue
		}

		inputs := getBrainInputs(e, rCount, snapBudget, snapSuccess, quadrantRadars, satCount, dayProgress)

		var idealAction []float64
		if k > 0 {
			idealAction = []float64{1.0, 0.0, 0.0} // Performance was good: stay/reinforce
		} else {
			idealAction = []float64{-1.0, 0.0, 0.0} // Performance was bad: relocate
		}

		wg.Add(1)
		jobs <- Job{
			Run: func() {
				defer wg.Done()
				// Use the updated Train signature (no channel passed in)
				brain.Train(inputs, idealAction, context)
			},
		}
	}

	// CRITICAL: Block until the GPU worker has processed every single radar.
	// Without this, the program 'hangs' or wipes stats before training is done.
	wg.Wait()

	// --- 3. Sync & Save Persistence ---
	// Now that training is done, sync weights to GPU and save the .gob file
	brain.mu.Lock()
	brain.SyncToGPU(context)
	brain.mu.Unlock()
	saveSystemState()

	// --- 4. Evaluate performance & bonuses ---
	if successRate >= 100.0 {
		difficultyMultiplier += 0.1
		EfficiencyBonus *= 1.1
	} else if successRate >= 99.5 {
		difficultyMultiplier += 0.05
		EfficiencyBonus *= 1.05
	} else if successRate >= 99.0 {
		EfficiencyBonus *= 1.025
	} else {
		EfficiencyBonus = 100000.0
	}

	// --- 5. Fleet pruning ---
	if successRate >= 95.0 {
		mu.RLock()
		sort.Slice(radarIDs, func(i, j int) bool {
			return kills[radarIDs[i]] < kills[radarIDs[j]]
		})
		mu.RUnlock()

		numToDrop := int(float64(rCount) * 0.15)
		dropped := 0
		now := simClock.Unix()
		gracePeriodSeconds := int64(6 * time.Hour.Seconds())

		for _, id := range radarIDs {
			if dropped >= numToDrop || (len(radarIDs)-dropped) <= MinRadars {
				break
			}

			mu.RLock()
			e, exists := entities[id]
			mu.RUnlock()
			if !exists || (now-e.StartTime) < gracePeriodSeconds {
				continue
			}

			mu.Lock()
			delete(entities, id)
			delete(kills, id)
			mu.Unlock()
			dropped++

			scalingFactor := 1.0 + (100.0 / float64(len(entities)-len(cityNames)+1))
			budget += EfficiencyBonus * scalingFactor
		}
	}

	// --- 6. Crisis recovery ---
	if budget < BankruptcyLimit || (rCount < MinRadars && budget < RadarCost) {
		log.Println("CRISIS: Adaptive Mutation Triggered...")
		brain.mu.Lock()
		brain.AdaptiveMutate(successRate, context)
		brain.mu.Unlock()
		budget = EmergencyBudget
	}

	lastEraSuccess = successRate

	// --- 7. Finalize era & Reset Stats ---
	mu.Lock()
	for id := range kills {
		kills[id] = 0 // Safe to clear now: training is 100% complete
	}
	currentCycle++

	eraStartTime = eraStartTime.Add(EraDuration)
	simClock = eraStartTime
	wallStart = time.Now()
	totalThreats, totalIntercepts = 0, 0
	for i := range quadrantMisses {
		quadrantMisses[i] = 0
	}
	mu.Unlock()

	log.Printf("--- ERA %d STARTED | PERSISTENCE MAINTAINED ---", currentCycle)
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
		city, exists := entities[name]
		// SAFETY CHECK: Ensure the city actually exists in the map
		if !exists || city == nil {
			continue
		}
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
		mu.RLock()
		lat, lon := getTetheredCoords()
		mu.RUnlock()
		return lat, lon
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

func gpuWorker(ctx *cuda.PrimaryCtx, jobs <-chan Job, ready chan<- struct{}) {
	// Lock this OS thread for the lifetime of the worker
	runtime.LockOSThread()

	// Set CUDA context
	if err := cuda.SetCurrentContext(&ctx.Context); err != nil {
		panic(err)
	}

	// Signal that GPU is ready
	if ready != nil {
		close(ready)
	}

	// Process jobs indefinitely until channel is closed
	for job := range jobs {
		job.Run()
	}

	// Worker exits; OS thread can be unlocked automatically
	// (optional in Go; usually left locked until process exit)
}

func runPhysicsEngine(ctx context.Context, cuda_ctx *cuda.PrimaryCtx, jobs chan<- Job) {
	var lastLaunch time.Time
	mu.Lock()
	if simClock.IsZero() {
		simClock = time.Now()
		eraStartTime = simClock
	}
	mu.Unlock()

	for {
		select {
		case <-ctx.Done():
			// This case triggers the moment cancel() is called elsewhere

			return
		default:
			if simClock.After(eraStartTime.Add(EraDuration)) || forceReset {
				forceReset = false
				autonomousEraReset(budget, successRate, difficultyMultiplier, cuda_ctx, jobs)
				continue
			}

			// Update Satellites
			satCount := 0
			mu.RLock()
			for _, e := range entities {
				if e.Type == "SAT" {
					satCount++
				}
			}
			mu.RUnlock()

			if satCount < MaxSatellites && simClock.Sub(lastLaunch) >= LaunchInterval {
				site := launchSites[rand.Intn(len(launchSites))]
				id := fmt.Sprintf("SAT-%d-%d", currentCycle, satCount)
				mu.Lock()
				entities[id] = &Entity{ID: id, Type: "SAT", LaunchLat: site.Lat, LaunchLon: site.Lon, Lat: site.Lat, Lon: site.Lon, StartTime: simClock.Unix()}
				mu.Unlock()
				lastLaunch = simClock
			}
			mu.RLock()
			for _, e := range entities {
				select {
				case <-ctx.Done():
					// This case triggers the moment cancel() is called elsewhere

					return
				default:
					if e.Type == "SAT" {
						e.Lat, e.Lon = getSatellitePos(e.StartTime, e.LaunchLat, e.LaunchLon, simClock)
					}
				}
			}
			mu.RUnlock()

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
			mu.Lock()
			for id, e := range entities {
				select {
				case <-ctx.Done():
					// This case triggers the moment cancel() is called elsewhere
					return
				default:
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
			}
			mu.Unlock()
			// Worker Pool
			var wg sync.WaitGroup
			numWorkers := runtime.NumCPU()
			batchSize := MissilesBatchedInFlight
			workerKills := make(chan map[string]int, numWorkers)
			workerStats := make(chan struct {
				ints, thrs int
				rew, pen   float64
				misses     []float64
			}, numWorkers)

			for w := 0; w < numWorkers; w++ {
				select {
				case <-ctx.Done():
					// This case triggers the moment cancel() is called elsewhere
					return
				default:
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
								if successRate < 85.0 || rCount < MinRadars {
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
			}

			go func() { wg.Wait(); close(workerKills); close(workerStats) }()

			for lk := range workerKills {
				select {
				case <-ctx.Done():
					// This case triggers the moment cancel() is called elsewhere
					return
				default:
					mu.Lock()
					for id, count := range lk {
						if _, ok := kills[id]; ok {
							kills[id] += count
						}
					}
					mu.Unlock()
				}
			}

			for ls := range workerStats {
				select {
				case <-ctx.Done():
					// This case triggers the moment cancel() is called elsewhere

					return
				default:
					totalIntercepts += ls.ints
					totalThreats += ls.thrs
					budget += (ls.rew - ls.pen)
					for i, v := range ls.misses {
						quadrantMisses[i] += v
					}
				}
			}

			if successRate < 99.0 {
				mutationRate = 0.01
			} else if successRate >= 99.0 && successRate < 99.9 {
				mutationRate = 0.001 // Reduce for stability
			} else if successRate >= 99.9 {
				mutationRate = 0.0001 // Lock in the "optimized" state
			}

			// 2. WIN STREAK & CONVERGENCE
			// Ensure a minimum threat volume before counting a 'Win' to prevent cheesing
			if successRate >= 100.0 && (totalThreats+totalIntercepts) > 50 {
				winStreakCounter++
				log.Printf("\n[!] VICTORY STREAK: %d/%d Eras at 100%% efficiency.\n", winStreakCounter, RequiredWinStreak)

				if winStreakCounter >= RequiredWinStreak {
					log.Println("\n==================================================")
					log.Println("CONVERGENCE REACHED: AEGIS SHIELD IS OPTIMIZED")
					mu.RLock()
					log.Printf("Final Fleet Size: %d | Total Eras: %d\n", len(entities)-len(cityNames), currentCycle)
					mu.RUnlock()
					log.Println("==================================================")

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
					log.Printf("\n[!] STREAK BROKEN: Efficiency fell to %.2f%%\n", successRate)
				}
				winStreakCounter = 0
			}

			if totalThreats+totalIntercepts > 0 {
				successRate = (float64(totalIntercepts) / float64(totalThreats+totalIntercepts)) * 100
			}

			if len(batchInputs) > 0 {
				// Capture slices locally to avoid closure issues
				inputs := append([][]float64(nil), batchInputs...)
				radars := append([]*Entity(nil), activeRadars...)

				job := Job{
					Run: func() {
						results := make([][]float64, len(inputs))
						for i, input := range inputs {
							// Evaluate on GPU
							results[i] = brain.Evaluate(input)
						}

						// Apply results
						for i, res := range results {
							e := radars[i]

							action := 0
							if res[0] > 0.1 {
								action = 1 // BUILD
							} else if res[0] < -0.1 {
								action = 2 // RELOCATE
							}

							latN, lonN := res[1], res[2]
							prec := math.Max(5.0, 20.0*(1.0-(successRate/100.0)))

							// Determine quadrant needing help
							targetQ := 0
							maxM := -1.0
							for qi, m := range quadrantMisses {
								if m > maxM {
									maxM = m
									targetQ = qi
								}
							}

							if action == 1 && rCount < MaxRadars && budget >= RadarCost {
								bLat, bLon := getTargetedTether(targetQ)
								id := fmt.Sprintf("R-AI-%d-%d", currentCycle, rand.Intn(1e7))

								entities[id] = &Entity{
									ID:        id,
									Type:      "RADAR",
									Lat:       bLat + (latN * prec / 111.0),
									Lon:       bLon + (lonN * prec / 111.0),
									StartTime: simClock.Unix(),
								}
								kills[id] = 0
								budget -= RadarCost
								rCount++

							} else if action == 2 && budget >= RelocationCost {
								bLat, bLon := getTargetedTether(targetQ)
								e.Lat = bLat + (latN * 50.0 / 111.0)
								e.Lon = bLon + (lonN * 50.0 / (111.0 * math.Cos(bLat*math.Pi/180.0)))
								e.LastMoved = time.Now().UnixMilli()
								kills[e.ID] = 0
								budget -= RelocationCost
							}
						}
					},
				}

				// Send to GPU worker
				gpuJobs <- job
			}
			mu.Lock()
			simClock = simClock.Add(TimeStepping)
			mu.Unlock()

			runtime.Gosched() // CRITICAL: Gives the web server/UI a window to get the lock
			//time.Sleep(10 * time.Millisecond) // Prevents 100% CPU lock
		}
	}
}

func getCityData() map[string][]float64 {
	// Extensive mapping of global population hubs for network optimization
	cityData := map[string][]float64{

		// --- Major Metropolitan Hubs ---
		"Toronto": {43.65, -79.38}, "Montreal": {45.50, -73.56}, "Vancouver": {49.28, -123.12},
		"Calgary": {51.04, -114.07}, "Edmonton": {53.54, -113.49}, "Ottawa": {45.42, -75.69},
		"Quebec City": {46.81, -71.20}, "Winnipeg": {49.89, -97.13}, "Hamilton": {43.25, -79.87},
		"Kitchener": {43.45, -80.51}, "London ON": {42.98, -81.24}, "Victoria": {48.42, -123.36},
		"Halifax": {44.64, -63.57}, "Oshawa": {43.89, -78.86}, "Windsor": {42.31, -83.03},
		"Saskatoon": {52.13, -106.67}, "Regina": {50.44, -104.61}, "St. John's": {47.56, -52.71},

		// --- Regional Hubs & Mid-Sized Cities ---
		"Sherbrooke": {45.40, -71.89}, "Trois-Rivieres": {46.34, -72.55}, "Gatineau": {45.43, -75.70},
		"Barrie": {44.39, -79.69}, "Kelowna": {49.88, -119.49}, "Abbotsford": {49.05, -122.33},
		"Guelph": {43.55, -80.23}, "Kingston": {44.23, -76.49}, "Sudbury": {46.49, -80.99},
		"Moncton": {46.09, -64.78}, "Saint John NB": {45.27, -66.06}, "Fredericton": {45.96, -66.64},
		"Charlottetown": {46.24, -63.13}, "Iqaluit": {63.75, -68.51},

		// --- Far North & Remote Regional Nodes ---
		"Whitehorse": {60.72, -135.05}, "Dawson City": {64.06, -139.43}, "Watson Lake": {60.07, -128.79},
		"Yellowknife": {62.45, -114.37}, "Hay River": {60.82, -115.79}, "Fort Smith": {60.00, -111.88},
		"Inuvik": {68.35, -133.72}, "Norman Wells": {65.28, -126.83},
		"Rankin Inlet": {62.81, -92.07}, "Cambridge Bay": {69.11, -105.05}, "Kugluktuk": {67.82, -115.11},
		"Kuujjuaq": {58.11, -68.40}, "Schefferville": {54.81, -66.83}, "Moosonee": {51.27, -80.64},
		"La Ronge": {55.10, -105.29}, "Prince George": {53.91, -122.75}, "Fort McMurray": {56.73, -111.38},
		"Thompson": {55.74, -97.85}, "Churchill": {58.77, -94.16}, "Gander": {48.96, -54.60},
		"Corner Brook": {48.95, -57.94}, "Happy Valley-Goose Bay": {53.30, -60.33},

		// --- NORTH AMERICA (Dense Regional Nodes) ---
		"New York": {40.71, -74.00}, "Los Angeles": {34.05, -118.24}, "Chicago": {41.87, -87.62},
		"Mexico City": {19.43, -99.13}, "Houston": {29.76, -95.36},
		"Miami": {25.76, -80.19}, "San Francisco": {37.77, -122.42}, "Seattle": {47.61, -122.33},
		"Dallas": {32.78, -96.80},
		"Boston": {42.36, -71.05}, "Atlanta": {33.75, -84.39}, "Denver": {39.74, -104.99},
		"Phoenix": {33.45, -112.07}, "Philadelphia": {39.95, -75.16}, "Washington": {38.89, -77.03},
		"San Diego": {32.72, -117.16}, "Austin": {30.26, -97.74}, "Indianapolis": {39.76, -86.15},
		"Charlotte": {35.22, -80.84}, "Nashville": {36.16, -86.78}, "Columbus": {39.96, -82.99},
		"Portland": {45.52, -122.67}, "Las Vegas": {36.17, -115.14}, "Detroit": {42.33, -83.04},
		"Memphis": {35.14, -90.04}, "Baltimore": {39.29, -76.61}, "Milwaukee": {43.03, -87.90},
		"Albuquerque": {35.08, -106.65}, "Tucson": {32.22, -110.97}, "Fresno": {36.73, -119.78},
		"Sacramento": {38.58, -121.49}, "Long Beach": {33.77, -118.19}, "Kansas City": {39.09, -94.57},
		"Mesa": {33.41, -111.83}, "Virginia Beach": {36.85, -75.97},
		"Omaha": {41.25, -95.93}, "Raleigh": {35.77, -78.63},
		"Cleveland": {41.49, -81.69}, "Tulsa": {36.15, -95.99}, "New Orleans": {29.95, -90.07},
		"Wichita": {37.68, -97.33}, "Tampa": {27.95, -82.45}, "Boise": {43.61, -116.20},
		"Quebec": {46.81, -71.20}, "St. Catharines": {43.15, -79.24},
		"Guadalajara": {20.67, -103.35}, "Monterrey": {25.68, -100.31},
		"Puebla": {19.04, -98.20}, "Tijuana": {32.51, -117.03}, "Leon": {21.12, -101.68},
		"Juarez": {31.73, -106.48}, "Zapopan": {20.72, -103.39}, "Queretaro": {20.58, -100.38},
		"San Luis Potosi": {22.15, -100.98}, "Merida": {20.96, -89.62}, "Mexicali": {32.62, -115.45},
		"Aguascalientes": {21.88, -102.29}, "Cuernavaca": {18.92, -99.22}, "Acapulco": {16.85, -99.88},
		"Cancun": {21.16, -86.85}, "Panama": {8.98, -79.52}, "San Jose": {9.93, -84.08},
		"Guatemala": {14.63, -90.50}, "San Salvador": {13.69, -89.21}, "Tegucigalpa": {14.07, -87.19},
		"Managua": {12.13, -86.23}, "Belmopan": {17.25, -88.76},

		// --- SOUTH AMERICA (Deep Mapping) ---
		"Sao Paulo": {-23.55, -46.63}, "Rio de Janeiro": {-22.90, -43.17}, "Buenos Aires": {-34.60, -58.38},
		"Lima": {-12.04, -77.04}, "Bogota": {4.71, -74.07}, "Santiago": {-33.44, -70.66},
		"Belo Horizonte": {-19.91, -43.93}, "Porto Alegre": {-30.03, -51.21}, "Brasilia": {-15.79, -47.88},
		"Fortaleza": {-3.71, -38.54}, "Salvador": {-12.97, -38.50}, "Recife": {-8.05, -34.88},
		"Curitiba": {-25.42, -49.27}, "Manaus": {-3.11, -60.02}, "Campinas": {-22.90, -47.06},
		"Belem": {-1.45, -48.49}, "Guarulhos": {-23.45, -46.53}, "Goiania": {-16.68, -49.25},
		"Sao Luis": {-2.53, -44.30}, "Maceio": {-9.66, -35.73}, "Natal": {-5.79, -35.21},
		"Teresina": {-5.09, -42.80}, "Joao Pessoa": {-7.11, -34.86}, "Aracaju": {-10.91, -37.07},
		"Cuiaba": {-15.60, -56.09}, "Florianopolis": {-27.59, -48.54}, "Londrina": {-23.30, -51.16},
		"Joinville": {-26.30, -48.85}, "Ribeirao Preto": {-21.17, -47.81}, "Uberlandia": {-18.91, -48.27},
		"Cordoba": {-31.42, -64.19}, "Rosario": {-32.94, -60.63}, "Mendoza": {-32.88, -68.84},
		"La Plata": {-34.92, -57.95}, "Tucuman": {-26.80, -65.22}, "Mar del Plata": {-38.00, -57.55},
		"Salta": {-24.78, -65.41}, "Santa Fe": {-31.63, -60.70}, "Medellin": {6.24, -75.58},
		"Cali": {3.45, -76.53}, "Barranquilla": {10.96, -74.79}, "Cartagena": {10.39, -75.48},
		"Bucaramanga": {7.12, -73.12}, "Pereira": {4.81, -75.69}, "Manizales": {5.06, -75.51},
		"Guayaquil": {-2.17, -79.92}, "Quito": {-0.18, -78.47}, "Cuenca": {-2.90, -79.00},
		"Santa Cruz": {-17.78, -63.18}, "La Paz": {-16.50, -68.15}, "Cochabamba": {-17.38, -66.15},
		"Caracas": {10.48, -66.90}, "Maracaibo": {10.65, -71.64}, "Valencia": {10.16, -68.00},
		"Montevideo": {-34.90, -56.16}, "Asuncion": {-25.26, -57.58}, "Paramaribo": {5.85, -55.20},
		"Cayenne": {4.93, -52.33}, "Georgetown": {6.80, -58.15}, "Valparaiso": {-33.04, -71.61},
		"Antofagasta": {-23.65, -70.40}, "Arequipa": {-16.40, -71.53}, "Trujillo": {-8.11, -79.03},
		"Chiclayo": {-6.77, -79.83}, "Iquitos": {-3.75, -73.25},

		// --- EUROPE (Extensive coverage) ---
		"London": {51.50, -0.12}, "Paris": {48.85, 2.35}, "Berlin": {52.52, 13.40},
		"Madrid": {40.41, -3.70}, "Rome": {41.90, 12.49}, "Moscow": {55.75, 37.61},
		"Istanbul": {41.00, 28.97}, "Barcelona": {41.38, 2.17}, "Milan": {45.46, 9.19},
		"Munich": {48.13, 11.58}, "Hamburg": {53.55, 9.99}, "Frankfurt": {50.11, 8.68},
		"Marseille": {43.29, 5.37}, "Lyon": {45.76, 4.83},
		"Toulouse": {43.60, 1.44}, "Nice": {43.71, 7.26}, "Nantes": {47.21, -1.55},
		"Strasbourg": {48.57, 7.75}, "Montpellier": {43.61, 3.87}, "Bordeaux": {44.83, -0.57},
		"Lille": {50.62, 3.06}, "Rennes": {48.11, -1.67}, "Birmingham": {52.48, -1.89},
		"Glasgow": {55.86, -4.25}, "Liverpool": {53.40, -2.99}, "Leeds": {53.80, -1.54},
		"Sheffield": {53.38, -1.47}, "Manchester": {53.48, -2.24}, "Edinburgh": {55.95, -3.18},
		"Bristol": {51.45, -2.58}, "Belfast": {54.59, -5.93}, "Dublin": {53.35, -6.26},
		"Naples": {40.85, 14.26}, "Turin": {45.07, 7.68}, "Palermo": {38.11, 13.36},
		"Genoa": {44.40, 8.94}, "Bologna": {44.49, 11.34}, "Florence": {43.76, 11.25},
		"Seville": {37.38, -5.98}, "Zaragoza": {41.64, -0.88},
		"Malaga": {36.72, -4.42}, "Murcia": {37.99, -1.13}, "Palma": {39.56, 2.65},
		"Cologne": {50.93, 6.96}, "Stuttgart": {48.77, 9.18}, "Dusseldorf": {51.22, 6.77},
		"Dortmund": {51.51, 7.46}, "Essen": {51.45, 7.01}, "Bremen": {53.07, 8.80},
		"Leipzig": {51.33, 12.37}, "Dresden": {51.05, 13.73}, "Hanover": {52.37, 9.73},
		"Amsterdam": {52.37, 4.90}, "Rotterdam": {51.92, 4.47}, "The Hague": {52.07, 4.30},
		"Utrecht": {52.09, 5.12}, "Eindhoven": {51.44, 5.47}, "Brussels": {50.85, 4.35},
		"Antwerp": {51.21, 4.40}, "Ghent": {51.05, 3.73}, "Vienna": {48.21, 16.37},
		"Graz": {47.07, 15.43}, "Prague": {50.08, 14.43}, "Brno": {49.19, 16.60},
		"Budapest": {47.50, 19.04}, "Warsaw": {52.22, 21.01}, "Krakow": {50.06, 19.94},
		"Lodz": {51.75, 19.45}, "Wroclaw": {51.10, 17.03}, "Poznan": {52.40, 16.90},
		"Gdansk": {54.35, 18.64}, "Stockholm": {59.32, 18.06}, "Gothenburg": {57.70, 11.97},
		"Malmo": {55.60, 13.00}, "Oslo": {59.91, 10.75}, "Bergen": {60.39, 5.32},
		"Trondheim": {63.43, 10.39}, "Copenhagen": {55.68, 12.57}, "Aarhus": {56.16, 10.21},
		"Helsinki": {60.17, 24.94}, "Espoo": {60.20, 24.65}, "Tampere": {61.49, 23.77},
		"Lisbon": {38.72, -9.14}, "Porto": {41.15, -8.61}, "Athens": {37.98, 23.73},
		"Thessaloniki": {40.64, 22.94}, "Bucharest": {44.42, 26.10}, "Cluj-Napoca": {46.77, 23.62},
		"Sofia": {42.69, 23.32}, "Belgrade": {44.78, 20.44}, "Zagreb": {45.81, 15.98},
		"Ljubljana": {46.05, 14.50}, "Sarajevo": {43.85, 18.43}, "Skopje": {41.99, 21.43},
		"Tirana": {41.32, 19.81}, "Podgorica": {42.44, 19.25}, "Chisinau": {47.01, 28.86},
		"Riga": {56.94, 24.10}, "Vilnius": {54.68, 25.27}, "Tallinn": {59.43, 24.75},
		"Minsk": {53.90, 27.56}, "Kyiv": {50.45, 30.52}, "Kharkiv": {49.99, 36.23},
		"Odesa": {46.48, 30.73}, "Dnipro": {48.46, 35.04}, "Saint Petersburg": {59.93, 30.33},
		"Novosibirsk": {55.03, 82.92}, "Yekaterinburg": {56.83, 60.60}, "Kazan": {55.78, 49.12},
		"Nizhny Novgorod": {56.32, 44.00}, "Chelyabinsk": {55.15, 61.40},

		// --- AFRICA (High Density Expansion) ---
		"Cairo": {30.04, 31.23}, "Lagos": {6.52, 3.37}, "Kinshasa": {-4.32, 15.32},
		"Johannesburg": {-26.20, 28.04}, "Alexandria": {31.20, 29.92}, "Abidjan": {5.35, -4.00},
		"Casablanca": {33.57, -7.58}, "Nairobi": {-1.29, 36.82}, "Cape Town": {-33.92, 18.42},
		"Durban": {-29.85, 31.02}, "Dar es Salaam": {-6.79, 39.20}, "Accra": {5.56, -0.20},
		"Algiers": {36.75, 3.06}, "Addis Ababa": {8.98, 38.79}, "Luanda": {-8.83, 13.23},
		"Kano": {12.00, 8.51}, "Ibadan": {7.37, 3.94}, "Douala": {4.05, 9.76},
		"Yaounde": {3.84, 11.50}, "Kumasi": {6.66, -1.61}, "Ouagadougou": {12.37, -1.53},
		"Bamako": {12.65, -8.00}, "Conakry": {9.53, -13.71}, "Freetown": {8.48, -13.23},
		"Monrovia": {6.30, -10.80}, "Dakar": {14.71, -17.46}, "Tripoli": {32.89, 13.18},
		"Tunis": {36.81, 10.18}, "Rabat": {34.02, -6.83}, "Marrakech": {31.62, -7.98},
		"Fes": {34.03, -5.00}, "Tangier": {35.77, -5.80}, "Kampala": {0.31, 32.58},
		"Kigali": {-1.94, 30.06}, "Bujumbura": {-3.38, 29.36}, "Lusaka": {-15.38, 28.32},
		"Harare": {-17.82, 31.05}, "Maputo": {-25.96, 32.57}, "Antananarivo": {-18.87, 47.50},
		"Lilongwe": {-13.96, 33.77}, "Blantyre": {-15.76, 35.00}, "Mombasa": {-4.04, 39.66},
		"Kisumu": {-0.09, 34.76}, "Mwanza": {-2.52, 32.90}, "Lubumbashi": {-11.68, 27.50},
		"Mbuji-Mayi": {-6.13, 23.58}, "Kisangani": {0.51, 25.20}, "Abuja": {9.07, 7.48},
		"Port Harcourt": {4.75, 7.00}, "Benin City": {6.33, 5.61}, "Brazzaville": {-4.26, 15.28},
		"Libreville": {0.39, 9.45}, "Bangui": {4.36, 18.57}, "N'Djamena": {12.11, 15.03},
		"Niamey": {13.51, 2.11}, "Nouakchott": {18.08, -15.97}, "Asmara": {15.32, 38.93},
		"Djibouti": {11.58, 43.14}, "Mogadishu": {2.04, 45.34},

		// --- ASIA (High Density & Tiers) ---
		"Tokyo": {35.68, 139.65}, "Osaka": {34.69, 135.50}, "Seoul": {37.56, 126.97},
		"Beijing": {39.90, 116.40}, "Shanghai": {31.23, 121.47}, "Mumbai": {19.07, 72.87},
		"Delhi": {28.61, 77.20}, "Karachi": {24.86, 67.01}, "Dhaka": {23.81, 90.41},
		"Jakarta": {-6.20, 106.81}, "Manila": {14.59, 120.98}, "Bangkok": {13.75, 100.50},
		"Ho Chi Minh": {10.82, 106.63}, "Guangzhou": {23.13, 113.26}, "Shenzhen": {22.54, 114.06},
		"Chongqing": {29.56, 106.55}, "Chengdu": {30.57, 104.06}, "Tianjin": {39.12, 117.19},
		"Wuhan": {30.59, 114.30}, "Hangzhou": {30.27, 120.15}, "Nanjing": {32.06, 118.79},
		"Xian": {34.34, 108.93}, "Suzhou": {31.29, 120.58}, "Qingdao": {36.06, 120.38},
		"Kolkata": {22.57, 88.36}, "Bangalore": {12.97, 77.59}, "Chennai": {13.08, 80.27},
		"Hyderabad": {17.38, 78.48}, "Ahmedabad": {23.02, 72.57}, "Pune": {18.52, 73.85},
		"Surat": {21.17, 72.83}, "Jaipur": {26.91, 75.78}, "Lucknow": {26.84, 80.94},
		"Nagpur": {21.14, 79.08}, "Indore": {22.71, 75.85}, "Bhopal": {23.25, 77.41},
		"Kanpur": {26.45, 80.33}, "Lahore": {31.55, 74.34}, "Faisalabad": {31.45, 73.13},
		"Rawalpindi": {33.60, 73.04}, "Multan": {30.15, 71.52}, "Chittagong": {22.35, 91.78},
		"Khulna": {22.84, 89.54}, "Singapore": {1.35, 103.81}, "Kuala Lumpur": {3.14, 101.69},
		"Klang": {3.04, 101.45}, "Johor Bahru": {1.49, 103.74}, "Bandung": {-6.91, 107.60},
		"Surabaya": {-7.25, 112.75}, "Medan": {3.59, 98.67}, "Bekasi": {-6.24, 106.99},
		"Tangerang": {-6.17, 106.63}, "Quezon City": {14.65, 121.05}, "Davao": {7.19, 125.45},
		"Cebu": {10.31, 123.88}, "Zamboanga": {6.90, 122.07}, "Hanoi": {21.03, 105.85},
		"Da Nang": {16.05, 108.27}, "Hai Phong": {20.85, 106.68}, "Yangon": {16.86, 96.19},
		"Mandalay": {21.95, 96.08}, "Phnom Penh": {11.55, 104.91}, "Vientiane": {17.97, 102.63},
		"Kyoto": {35.01, 135.76}, "Sapporo": {43.06, 141.35}, "Fukuoka": {33.59, 130.40},
		"Nagoya": {35.18, 136.90}, "Hiroshima": {34.38, 132.45}, "Sendai": {38.26, 140.87},
		"Incheon": {37.45, 126.70}, "Daegu": {35.87, 128.60}, "Gwangju": {35.15, 126.85},
		"Daejeon": {36.35, 127.38}, "Taipei": {25.03, 121.56}, "Kaohsiung": {22.62, 120.31},

		// --- MIDDLE EAST & WEST ASIA ---
		"Tehran": {35.68, 51.38}, "Riyadh": {24.71, 46.67},
		"Baghdad": {33.31, 44.36}, "Dubai": {25.20, 55.27}, "Jeddah": {21.49, 39.19},
		"Ankara": {39.93, 32.85}, "Izmir": {38.42, 27.14}, "Amman": {31.95, 35.93},
		"Damascus": {33.51, 36.27}, "Aleppo": {36.20, 37.13}, "Beirut": {33.89, 35.50},
		"Tel Aviv": {32.08, 34.78}, "Jerusalem": {31.77, 35.21}, "Doha": {25.29, 51.53},
		"Kuwait": {29.38, 47.98}, "Muscat": {23.58, 58.40}, "Abu Dhabi": {24.45, 54.37},
		"Sharjah": {25.34, 55.41}, "Manama": {26.22, 50.58}, "Sanaa": {15.36, 44.19},
		"Aden": {12.78, 45.01}, "Tabriz": {38.08, 46.29}, "Isfahan": {32.65, 51.66},
		"Mashhad": {36.29, 59.60}, "Shiraz": {29.59, 52.53}, "Baku": {40.41, 49.87},
		"Tbilisi": {41.72, 44.83}, "Yerevan": {40.18, 44.51}, "Kabul": {34.53, 69.17},
		"Kandahar": {31.61, 65.71}, "Tashkent": {41.29, 69.24}, "Almaty": {43.24, 76.88},
		"Astana": {51.16, 71.47}, "Bishkek": {42.87, 74.59}, "Dushanbe": {38.53, 68.78},
		"Ashgabat": {37.96, 58.32}, "Ulaanbaatar": {47.88, 106.89},

		// --- OCEANIA (Dense Regional Nodes) ---
		"Sydney": {-33.86, 151.20}, "Melbourne": {-37.81, 144.96}, "Brisbane": {-27.47, 153.03},
		"Perth": {-31.95, 115.86}, "Adelaide": {-34.92, 138.60}, "Auckland": {-36.84, 174.76},
		"Wellington": {-41.29, 174.78}, "Christchurch": {-43.53, 172.63}, "Canberra": {-35.28, 149.13},
		"Hobart": {-42.88, 147.32}, "Darwin": {-12.46, 130.84}, "Gold Coast": {-28.01, 153.40},
		"Newcastle": {-32.92, 151.77}, "Cairns": {-16.92, 145.77}, "Port Moresby": {-9.44, 147.18},
		"Suva": {-18.14, 178.44}, "Noumea": {-22.27, 166.44}, "Papeete": {-17.53, -149.57},
	}
	return cityData
}

func initialisation() *cuda.InitToken {
	dev, err := cuda.Init(0) // 0 is the device number
	if err != nil {
		panic(err) // or handle the error
	}
	return dev // close the library when done. REQUIRED, see below
}

func main() {

	// -------------------------------------------------
	// 1. INITIALIZE CUDA DEVICE + CONTEXT
	// -------------------------------------------------
	dev := initialisation()
	defer dev.Close()

	// -------------------------------------------------
	// 2. INITIALIZE NEURAL NETWORK (GPU READY)
	// MUST happen AFTER CUDA init
	// -------------------------------------------------
	brain = NewCudaBrain()

	// -------------------------------------------------
	// 3. CREATE GPU JOB QUEUE + WORKER
	// Worker owns CUDA context + locked OS thread
	// -------------------------------------------------
	gpuJobs = make(chan Job, 64)

	ready := make(chan struct{})
	go gpuWorker(dev.PrimaryCtx, gpuJobs, ready)
	<-ready

	// -------------------------------------------------
	// 4. SYSTEM RUNTIME CONFIG
	// -------------------------------------------------
	log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds)
	runtime.GOMAXPROCS(runtime.NumCPU())

	// -------------------------------------------------
	// 5. POPULATE CITY DATABASE
	// -------------------------------------------------
	cityData := getCityData()

	for name, pos := range cityData {
		entities[name] = &Entity{
			ID:   name,
			Type: "CITY",
			Lat:  pos[0],
			Lon:  pos[1],
		}
		cityNames = append(cityNames, name)
	}

	log.Printf("Protecting %d cities\n", len(cityData))

	// -------------------------------------------------
	// 6. LOAD PERSISTENCE
	// (May overwrite brain weights)
	// -------------------------------------------------
	loadSystemState(dev.PrimaryCtx)

	// -------------------------------------------------
	// 7. SEED RADAR FLEET IF NEEDED
	// -------------------------------------------------
	radarCount := 0
	for _, e := range entities {
		if e.Type == "RADAR" {
			radarCount++
		}
	}

	if radarCount < MinRadars {

		log.Println("NO PERSISTENCE FOUND. INITIALIZING SEED FLEET...")

		for i := 0; i < MinRadars; i++ {

			id := fmt.Sprintf("R-START-%d", i)
			lat, lon := getTetheredCoords()

			entities[id] = &Entity{
				ID:        id,
				Type:      "RADAR",
				Lat:       lat,
				Lon:       lon,
				StartTime: time.Now().Unix(),
			}
		}

	} else {
		log.Printf("RESUMING ERA %d WITH %d OPTIMIZED ASSETS\n", currentCycle, radarCount)
	}

	// -------------------------------------------------
	// 8. START SIMULATION CLOCK
	// -------------------------------------------------
	wallStart = time.Now()

	// -------------------------------------------------
	// 9. START PHYSICS ENGINE
	// -------------------------------------------------
	go runPhysicsEngine(ctx, dev.PrimaryCtx, gpuJobs)

	// -------------------------------------------------
	// 10. START MANUAL / WEB CONTROL SERVER
	// -------------------------------------------------
	go startManualServer(dev.PrimaryCtx)

	log.Println("AEGIS RUNNING AT :8080")

	// -------------------------------------------------
	// 11. GRACEFUL SHUTDOWN HANDLER
	// -------------------------------------------------
	sigChan := make(chan os.Signal, 1)

	signal.Notify(sigChan,
		syscall.SIGINT,
		syscall.SIGTERM,
	)

	<-sigChan

	log.Println("Shutdown signal received")

	cancel()          // stop physics engine
	close(gpuJobs)    // stop GPU worker
	saveSystemState() // persist fleet + brain

	log.Println("Shutdown complete")
}

func sendHTTPResponse(conn net.Conn, contentType string, data []byte) {
	header := fmt.Sprintf(
		"HTTP/1.1 200 OK\r\n"+
			"Content-Type: %s\r\n"+
			"Content-Length: %d\r\n"+
			"Connection: close\r\n\r\n",
		contentType, len(data),
	)
	conn.Write([]byte(header))
	conn.Write(data)
}

func startManualServer(context *cuda.PrimaryCtx) {
	listener, err := net.Listen("tcp", ":8080")
	if err != nil {
		return
	}

	for {
		conn, err := listener.Accept()
		if err != nil {
			continue
		}
		// Handle each connection in a way that respects your Mutex
		go handleConnection(conn, context)
	}
}

func handleConnection(conn net.Conn, ctx *cuda.PrimaryCtx) {
	defer conn.Close()

	buf := make([]byte, 1024)
	n, err := conn.Read(buf)
	if err != nil {
		return
	}

	// Use bytes.Contains for the check
	if bytes.Contains(buf[:n], []byte("GET /intel")) {
		// Prepare Intel Data
		serveIntel(conn)
	} else if bytes.Contains(buf[:n], []byte("GET /panic")) {
		servePanic(conn, ctx)
		sendHTTPResponse(conn, "text/html", []byte(uiHTML))
	} else if bytes.Contains(buf[:n], []byte("GET /save")) {
		saveSystemState()
		sendHTTPResponse(conn, "text/html", []byte(uiHTML))
	} else {
		sendHTTPResponse(conn, "text/html", []byte(uiHTML))
	}
}

func servePanic(conn net.Conn, context *cuda.PrimaryCtx) {
	cancel()
	time.Sleep(10 * time.Millisecond) // to let the cancel signal time to reach the goroutines.
	if mu.TryLock() {
		entities = make(map[string]*Entity)
		kills = make(map[string]int)
		citydata := getCityData()
		for name, pos := range citydata {
			entities[name] = &Entity{ID: name, Type: "CITY", Lat: pos[0], Lon: pos[1]}
		}
		for i := 0; i < MinRadars; i++ {
			id := fmt.Sprintf("R-START-%d", i)
			lat, lon := getTetheredCoords()
			entities[id] = &Entity{ID: id, Type: "RADAR", Lat: lat, Lon: lon, StartTime: time.Now().Unix()}
		}
		forceReset = true
		mu.Unlock()
	} else {
		if !mu.TryLock() {
			mu.Unlock()
			entities = make(map[string]*Entity)
			kills = make(map[string]int)
			citydata := getCityData()
			for name, pos := range citydata {
				entities[name] = &Entity{ID: name, Type: "CITY", Lat: pos[0], Lon: pos[1]}
			}
			for i := 0; i < MinRadars; i++ {
				id := fmt.Sprintf("R-START-%d", i)
				lat, lon := getTetheredCoords()
				entities[id] = &Entity{ID: id, Type: "RADAR", Lat: lat, Lon: lon, StartTime: time.Now().Unix()}
			}
			forceReset = true
		} else {
			entities = make(map[string]*Entity)
			kills = make(map[string]int)
			citydata := getCityData()
			for name, pos := range citydata {
				entities[name] = &Entity{ID: name, Type: "CITY", Lat: pos[0], Lon: pos[1]}
			}
			for i := 0; i < MinRadars; i++ {
				id := fmt.Sprintf("R-START-%d", i)
				lat, lon := getTetheredCoords()
				entities[id] = &Entity{ID: id, Type: "RADAR", Lat: lat, Lon: lon, StartTime: time.Now().Unix()}
			}
			forceReset = true
			mu.Unlock()
		}
	}
	go runPhysicsEngine(ctx, context, gpuJobs)
}

func serveIntel(conn net.Conn) {
	// Calculate Years Per Second
	realSeconds := time.Since(wallStart).Seconds()
	mu.RLock()
	simHours := simClock.Sub(eraStartTime).Hours() + (float64(currentCycle-1) * EraDuration.Hours())
	mu.RUnlock()
	simYears := simHours / (24 * 365)

	yps := 0.0
	if realSeconds > 0 {
		yps = simYears / realSeconds
	}

	mu.RLock()
	var all []Entity
	for _, e := range entities {
		all = append(all, *e)
	}
	mu.RUnlock()

	json.NewEncoder(conn).Encode(map[string]interface{}{
		"cycle":         currentCycle,
		"budget":        budget,
		"entities":      all,
		"success":       successRate,
		"streak":        winStreakCounter,
		"isOver":        isSimulationOver,
		"yps":           yps,
		"max_radars":    MaxRadars,
		"mutation_rate": mutationRate,
		"server_time":   time.Now().UnixMilli(),
	})
}

const uiHTML = `
<!DOCTYPE html><html><head><title>AEGIS REAL-TIME MONITOR</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
    body { margin:0; background:#000810; color:#0f0; font-family:'Segoe UI', 'Courier New', monospace; overflow:hidden; } 
    #stats { 
        position:fixed; top:15px; left:15px; z-index:1000; 
        background:rgba(0,10,20,0.9); padding:20px; border:1px solid #0af; 
        box-shadow: 0 0 15px rgba(0,170,255,0.3); border-radius: 8px; width: 320px;
        backdrop-filter: blur(5px);
    } 
    #map { height:100vh; width:100vw; background: #000; }
    .stat-line { font-size: 0.95em; margin-bottom: 8px; display: flex; justify-content: space-between; border-bottom: 1px solid #0af2; padding-bottom: 4px; }
    .val { color: #fff; font-weight: bold; }
    
    /* Confidence & Intelligence Dashboard */
    .intel-box { 
        margin-top: 15px; padding: 12px; border: 1px solid #fb0; 
        background: rgba(251, 176, 0, 0.1); text-align: center; border-radius: 4px;
    }
    #confidence-bar { height: 6px; background: #111; margin-top: 10px; border-radius: 3px; border: 1px solid #0af3; overflow: hidden; }
    #confidence-fill { height: 100%; width: 0%; transition: 0.8s cubic-bezier(0.4, 0, 0.2, 1); background: #fb0; }
    #ai-level { font-weight: bold; font-size: 0.85em; display: block; margin-top: 5px; text-transform: uppercase; letter-spacing: 1px; }
    #lr-val { font-size: 0.75em; color: #aaa; margin-top: 4px; display: block; }

    /* Control Buttons */
    .btn-group { display: flex; flex-direction: column; gap: 8px; margin-top: 15px; }
    .ctrl-btn {
        width: 100%; padding: 10px; font-family: inherit; font-weight: bold;
        cursor: pointer; transition: 0.2s; border-radius: 4px; background: transparent;
    }
    #save-btn { border: 1px solid #0af; color: #0af; }
    #save-btn:hover { background: #0af; color: #000; box-shadow: 0 0 10px #0af; }
    #panic-btn { border: 1px solid #f44; color: #f44; background: rgba(100,0,0,0.1); }
    #panic-btn:hover { background: #f44; color: #000; box-shadow: 0 0 10px #f44; }
</style></head>
<body>
<div id="stats">
    <div style="font-size:1.2em; font-weight:bold; color:#0af; margin-bottom:15px; border-bottom:2px solid #0af;">AEGIS CONTROL STRATUM</div>
    <div class="stat-line">SYSTEM STATUS <span id="status" class="val" style="color:#0f0">OPERATIONAL</span></div>
    <div class="stat-line">ERA <span id="era" class="val">0</span></div>
    <div class="stat-line">FLEET SIZE <span id="rcount" class="val">0</span> / <span id="max_radars">0</span></div>
    <div class="stat-line">EFFICIENCY <span id="success" class="val">0.00</span>%</div>
    <div class="stat-line">THROUGHPUT <span id="yps" class="val">0</span> Y/sec</div>
    <div class="stat-line">BUDGET <span id="budget" class="val" style="color:#fb0">$0</span></div>
    
    <div class="intel-box">
        <span style="font-size:0.7em; color:#aaa; font-weight: bold;">AI COGNITION LEVEL</span>
        <span id="ai-level">CALIBRATING...</span>
        <div id="confidence-bar"><div id="confidence-fill"></div></div>
        <span id="lr-val">MUTATION RATE: 0.00000</span>
    </div>

    <div class="btn-group">
        <button id="save-btn" class="ctrl-btn" onclick="saveRadarFile()">SAVE OPTIMIZED RADAR STATE</button>
        <button id="panic-btn" class="ctrl-btn" onclick="triggerPanic()">INITIATE SYSTEM PURGE</button>
    </div>
</div>
<div id="map"></div>

<script>
    var map = L.map('map', { zoomControl:false, attributionControl:false, preferCanvas: true }).setView([20, 0], 2);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png').addTo(map);
    
    var layers = {};
    var isFetching = false;

    function getAILogic(yps, success) {
        // Incremental Levels of Confidence vs. Throughput
        if (success > 99.8 && yps < 0.0001) return { label: "OVER-CONFIDENT", color: "#0af", width: 100 };
        if (success > 99) return { label: "STEADY STATE", color: "#0f0", width: 85 };
        if (success > 95) return { label: "REINFORCING PATTERNS", color: "#fb0", width: 65 };
        if (yps > 0.05) return { label: "HEURISTIC DISCOVERY", color: "#f70", width: 45 };
        if (yps > 0.01) return { label: "AGGRESSIVE LEARNING", color: "#f44", width: 30 };
        return { label: "INITIAL EXPLORATION", color: "#aaa", width: 15 };
    }

    async function saveRadarFile() {
        const btn = document.getElementById('save-btn');
        btn.innerText = "WRITING TO DISK...";
        try { 
            const resp = await fetch('/save'); 
            if(resp.ok) {
                btn.innerText = "RADAR.JSON SECURED";
                btn.style.color = "#0f0";
                setTimeout(() => { btn.innerText = "SAVE OPTIMIZED RADAR STATE"; btn.style.color = "#0af"; }, 2000);
            }
        } catch (e) { btn.innerText = "SAVE FAILED"; }
    }

    async function triggerPanic() {
        if(!confirm("WIPE NEURAL NETWORK WEIGHTS? This resets the learning streak.")) return;
        document.getElementById('status').innerText = "PURGING...";
        document.getElementById('status').style.color = "#f44";
        try { await fetch('/panic'); } catch (e) { console.error(e); }
		location.reload(true);
    }

    async function updateUI() {
        if (isFetching) return;
        isFetching = true;

        try {
            const response = await fetch('/intel');
            const data = await response.json();

            // 1. Dashboard Sync (Restored All Fields)
            document.getElementById('era').innerText = data.cycle || 0;
            document.getElementById('success').innerText = (data.success || 0).toFixed(2);
            document.getElementById('budget').innerText = "$" + Math.floor(data.budget || 0).toLocaleString();
            document.getElementById('yps').innerText = (data.yps || 0).toFixed(6);
            document.getElementById('max_radars').innerText = data.max_radars || 120;
            document.getElementById('lr-val').innerText = "MUTATION: " + (data.mutation_rate || 0).toFixed(5);

            // 2. Confidence & Throughput Logic
            const ai = getAILogic(data.yps || 0, data.success || 0);
            const levelEl = document.getElementById('ai-level');
            levelEl.innerText = ai.label;
            levelEl.style.color = ai.color;
            const fill = document.getElementById('confidence-fill');
            fill.style.width = ai.width + "%";
            fill.style.background = ai.color;

            // 3. Map Rendering (Restored Pulse & Cleanup)
            const now = data.server_time || Date.now(); 
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
                    const movedRecently = e.last_moved && (now - e.last_moved < 2000);
                    if (movedRecently) {
                        layers[e.id].setStyle({ color: '#ff0', weight: 5, fillOpacity: 0.5 });
                    } else if (e.type === 'RADAR') {
                        layers[e.id].setStyle({ color: '#0f0', weight: 1, fillOpacity: 0.1 });
                    }
                }
            });

            document.getElementById('rcount').innerText = radarCount;

            for (let id in layers) {
                if (!currentIds.has(id)) { map.removeLayer(layers[id]); delete layers[id]; }
            }
        } catch (e) { console.error("UI Data Sync Failure:", e); }
        isFetching = false;
    }
    
    setInterval(updateUI, 43); // 23 FPS Sync
</script></body></html>`

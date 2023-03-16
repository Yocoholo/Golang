package main

import (
	"fmt"
	"math/rand"
	"nnl/src/neuralnetwork"
	"nnl/src/trainingdata"
	"time"
)

func main() {
	defer timeTrack(time.Now(), "main")

	nn := neuralnetwork.SetUp(2, 4, 1, 0.1)
	trainingdata := getTrainingData()
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < 10000; i++ {
		index := rand.Intn(len(trainingdata))
		data := trainingdata[index]
		neuralnetwork.Train(&nn, data.Inputs, data.Targets)
	}
	errorCount := 0
	equalCount := 0
	epoch := 100000
	for i := 0; i < epoch; i++ {
		if tester(neuralnetwork.Feed(&nn, []float32{1, 0})) {
			equalCount++
		} else {
			errorCount++
		}
		if tester(neuralnetwork.Feed(&nn, []float32{0, 1})) {
			equalCount++
		} else {
			errorCount++
		}
		if tester(neuralnetwork.Feed(&nn, []float32{0, 0})) {
			errorCount++
		} else {
			equalCount++
		}
		if tester(neuralnetwork.Feed(&nn, []float32{1, 1})) {
			errorCount++
		} else {
			equalCount++
		}
	}
	fmt.Println("errorCount is:", errorCount)
	fmt.Println("equalsCount is:", equalCount)
	fmt.Println("iterating over", epoch*4, "times the nn is correct", percentage(epoch, equalCount), "% of the time")
}

func timeTrack(start time.Time, name string) {
	fmt.Printf("%v: %v\n", name, time.Since(start))
}

func percentage(epoc, equalCount int) float32 {
	x := float32(epoc * 4)
	y := float32(equalCount)
	z := y / x
	return z * 100
}

func tester(m []float32) bool {
	return m[0] >= 0.5
}

func getTrainingData() []trainingdata.Trainingdata {
	return []trainingdata.Trainingdata{
		{
			Inputs:  []float32{0, 1},
			Targets: []float32{1},
		},
		{
			Inputs:  []float32{1, 0},
			Targets: []float32{1},
		},
		{
			Inputs:  []float32{0, 0},
			Targets: []float32{0},
		},
		{
			Inputs:  []float32{1, 1},
			Targets: []float32{0},
		},
	}

}

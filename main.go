package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
)

type result struct {
	name   string
	points [][3]float64
}

func calcFunction(x1, x2 float64) float64 {
	return 2*math.Pow(x1, 3) + x1*math.Pow(x2, 2) - 216*x1
}

func firstDerivativeX1(x1, x2 float64) float64 {
	return 6*math.Pow(x1, 2) + math.Pow(x2, 2) - 216
}

func firstDerivativeX2(x1, x2 float64) float64 {
	return 2 * x1 * x2
}

func secondDerivativeX1(x1, x2 float64) float64 {
	return 12 * x1
}

func secondDerivativeX2(x1, x2 float64) float64 {
	return 2 * x1
}

func isAccurate(x1, x2, e float64) bool {
	return math.Abs(firstDerivativeX1(x1, x2)) <= e && math.Abs(firstDerivativeX2(x1, x2)) <= e
}

func main() {
	fmt.Println("Current function: 2*x1^3 + x1*x2^2 - 216*x1")
	x1, x2, e, err := readUserInput()
	if err != nil {
		fmt.Println(err)
		os.Exit(0)
	}

	gradientReceiver := make(chan result)
	coordinateDescentReceiver := make(chan result)
	newtontReceiver := make(chan result)
	points := [][3]float64{{x1, x2, calcFunction(x1, x2)}}
	go gradientMethod(e, 1.0, points, gradientReceiver, 1000)
	go coordinateDescentMethod(e, points, coordinateDescentReceiver, 1000)
	go newtonMethod(e, points, newtontReceiver, 1000)

	for runtime.NumGoroutine() != 1 {
		select {
		case result := <-gradientReceiver:
			getBestPoint("Gradient method", result)
			writePoints(fmt.Sprintf("Gradient_method%s", result.name), result.points)
		case result := <-coordinateDescentReceiver:
			getBestPoint("Coordinate Descent method", result)
			writePoints(fmt.Sprintf("Coordinate_Descent%s", result.name), result.points)
		case result := <-newtontReceiver:
			getBestPoint("Newton method", result)
			writePoints(fmt.Sprintf("Newton%s", result.name), result.points)
		}
	}
}

func gradientMethod(e, a float64, points [][3]float64, receiver chan result, maxIter int) {
	innerA := a
	x1, x2, f := points[0][0], points[0][1], points[0][2]

	for k := 0; (k < maxIter) && !isAccurate(points[k][0], points[k][1], e); k++ {
		for i := 0; (i < maxIter) && (f >= points[k][2]); i++ {
			x1 = points[k][0] - innerA*firstDerivativeX1(points[k][0], points[k][1])
			x2 = points[k][1] - innerA*firstDerivativeX2(points[k][0], points[k][1])
			f = calcFunction(x1, x2)
			innerA = innerA / 2
		}
		innerA = a
		points = append(points, [3]float64{x1, x2, f})
	}
	receiver <- result{
		name:   fmt.Sprintf("(%v,%v)", points[0][0], points[0][1]),
		points: points}
}

func coordinateDescentMethod(e float64, points [][3]float64, receiver chan result, maxIter int) {
	x1, x2 := points[0][0], points[0][1]

	for k := 0; (k < maxIter) && !isAccurate(points[k][0], points[k][1], e); k++ {
		x1Der := firstDerivativeX1(points[k][0], points[k][1])
		if math.Abs(x1Der) > e {
			x1 = points[k][0] - (x1Der / secondDerivativeX1(points[k][0], points[k][1]))
		}

		x2Der := firstDerivativeX2(points[k][0], points[k][1])
		if math.Abs(x2Der) > e {
			x2 = points[k][1] - (x2Der / secondDerivativeX2(points[k][0], points[k][1]))
		}
		points = append(points, [3]float64{x1, x2, calcFunction(x1, x2)})
	}
	receiver <- result{
		name:   fmt.Sprintf("(%v,%v)", points[0][0], points[0][1]),
		points: points}
}

func newtonMethod(e float64, points [][3]float64, receiver chan result, maxIter int) {
	for k := 0; (k < maxIter) && !isAccurate(points[k][0], points[k][1], e); k++ {
		x1 := points[k][0] - math.Pow(secondDerivativeX1(points[k][0], points[k][1]), -1)*firstDerivativeX1(points[k][0], points[k][1])
		x2 := points[k][1] - math.Pow(secondDerivativeX2(points[k][0], points[k][1]), -1)*firstDerivativeX2(points[k][0], points[k][1])
		points = append(points, [3]float64{x1, x2, calcFunction(x1, x2)})
	}
	receiver <- result{
		name:   fmt.Sprintf("(%v,%v)", points[0][0], points[0][1]),
		points: points}
}

func readUserInput() (float64, float64, float64, error) {
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Enter X1: ")
	x1, _ := reader.ReadString('\n')
	fmt.Print("Enter X2: ")
	x2, _ := reader.ReadString('\n')
	fmt.Print("Enter Accuracy: ")
	e, _ := reader.ReadString('\n')

	x1f, err1 := strconv.ParseFloat(strings.Replace(x1, "\r\n", "", -1), 64)
	x2f, err2 := strconv.ParseFloat(strings.Replace(x2, "\r\n", "", -1), 64)
	ef, err3 := strconv.ParseFloat(strings.Replace(e, "\r\n", "", -1), 64)

	if err1 != nil || err2 != nil || err3 != nil {
		return 0, 0, 0, fmt.Errorf("wrong input")
	}
	return x1f, x2f, ef, nil
}

func getBestPoint(methodName string, result result) {
	fmt.Printf("----Best results of %s-----\n", methodName)
	fmt.Printf("%s with starting point%s X1 and X2: (%v, %v)\n", methodName, result.name, result.points[len(result.points)-1][0], result.points[len(result.points)-1][1])
	fmt.Printf("%s with starting point%s F: %g\n", methodName, result.name, result.points[len(result.points)-1][2])
	fmt.Printf("%s with starting point%s K: %v\n", methodName, result.name, len(result.points)-1)
}

func writePoints(fileName string, points [][3]float64) {
	stringPoints := ""
	fullFileName := fmt.Sprintf("output/%s.txt", fileName)
	for _, p := range points {
		stringPoints = fmt.Sprintf("%s(%v, %v) | %v\n", stringPoints, p[0], p[1], p[2])
	}
	err := os.WriteFile(fullFileName, []byte(stringPoints), 0777)
	if err != nil {
		fmt.Println("Error while was writing into a file")
		return
	}
	fmt.Printf("All intermediate results have been successfully written to %s\n\n", fullFileName)
}

package matrix

import (
	"fmt"
	"math/rand"
	"os"
	"time"
)

// Matrix is a struct that holds rows, cols int and data [][]float32
type Matrix struct {
	Rows, Cols int
	Data       [][]float32
}

// SetUp is a 'constructor'
func SetUp(rows int, cols int) Matrix {
	return Matrix{
		Rows: rows,
		Cols: cols,
		Data: setUpData(rows, cols),
	}
}

// setUpData is initializes the data to [rows][cols]
func setUpData(rows int, cols int) [][]float32 {
	var tmpMatrix = make([][]float32, rows)
	for x := 0; x < rows; x++ {
		tmpMatrix[x] = make([]float32, cols)
	}
	return tmpMatrix
}

// ToArray matrix data to []float32
func ToArray(data Matrix) []float32 {
	d := make([]float32, data.Rows)
	for i := 0; i < data.Rows; i++ {
		d[i] = data.Data[i][0]
	}
	return d
}

// FromArray sets the data within the matrix to the float array that is passed
func FromArray(data []float32) Matrix {
	d := SetUp(len(data), 1)
	for i := 0; i < d.Rows; i++ {
		d.Data[i][0] = data[i]
	}
	return d
}

type fn func(float32) float32

// Map apply a func to every element of data
func Map(sig fn, matrix Matrix) Matrix {
	m := matrix
	for x := 0; x < m.Rows; x++ {
		for y := 0; y < m.Cols; y++ {
			m.Data[x][y] = sig(m.Data[x][y])
		}
	}
	return m
}

type operation func(float32, float32) float32

// Scalar applies operation to A.data[x][y] by n float32
func Scalar(m Matrix, op operation, n float32) Matrix {
	for x := 0; x < m.Rows; x++ {
		for y := 0; y < m.Cols; y++ {
			m.Data[x][y] = op(n, m.Data[x][y])
		}
	}
	return m
}

// ScalarMatrix applies operation to A.data[x][y] by B.data[x][y]
func ScalarMatrix(m1 Matrix, op operation, m2 Matrix) Matrix {
	m3 := m1
	if m1.Rows == m2.Rows && m1.Cols == m2.Cols {
		for x := 0; x < m1.Rows; x++ {
			for y := 0; y < m1.Cols; y++ {
				m3.Data[x][y] = op(m1.Data[x][y], m2.Data[x][y])
			}
		}
	} else {
		fmt.Println("matrix demensions are not equal for passed matrices", "\n", m1, "\n", m2)
		os.Exit(0)
	}
	return m3
}

// DotMatrix multiplies A.data[x][y] by B.data[x][y]
func DotMatrix(a Matrix, b Matrix) Matrix {
	if a.Cols != b.Rows {
		fmt.Println("m1 cols need to equal m2 rows", "\n", a, "\n", b)
		os.Exit(0)
	}
	m3 := SetUp(a.Rows, b.Cols)
	for x := 0; x < a.Rows; x++ {
		for y := 0; y < b.Cols; y++ {
			sum := float32(0)
			for z := 0; z < a.Cols; z++ {
				sum += a.Data[x][z] * b.Data[z][y]
			}
			m3.Data[x][y] = sum
		}
	}
	return m3
}

// Transpose takes a matrix and sets all rows to cols and all cols to rows
func Transpose(m1 Matrix) Matrix {
	transposeM1 := SetUp(m1.Cols, m1.Rows)
	for x := 0; x < m1.Rows; x++ {
		for y := 0; y < m1.Cols; y++ {
			transposeM1.Data[y][x] = m1.Data[x][y]
		}
	}
	return transposeM1
}

// Randomize takes a matrix and randomises all data points
func Randomize(m *Matrix) {
	for x := 0; x < m.Rows; x++ {
		rand.Seed(time.Now().UnixNano())
		for y := 0; y < m.Cols; y++ {
			m.Data[x][y] = float32(rand.Float32()*2 - 1)
		}
	}
}

// Add returns x + y
func Add(x, y float32) float32 {
	return x + y
}

// Subtract returns x - y
func Subtract(x, y float32) float32 {
	return x - y
}

// Multiply returns x * y
func Multiply(x, y float32) float32 {
	return x * y
}

// Divide returns x / y
func Divide(x, y float32) float32 {
	return x / y
}

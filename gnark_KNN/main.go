package main

import (
	"fmt"
	"math/big"

	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

// -----------------------------------------
// CONFIG — tweak freely for benchmarking
// -----------------------------------------
// N controls the dimensionality of the kernel / covariance matrix.
const N = 4
// Trials controls how many Freivalds rounds we run for the inverse check.
const Trials = 2

// Circuit enforces the ridge-regression complexity certificate:
// 1) B = K + lambda * I
// 2) BInv is (probabilistically) the inverse of B checked via Freivalds
// 3) complexity = tr(K * BInv)
//
// Public inputs:
//   - K        : kernel matrix (N x N)
//   - Lambda   : ridge parameter
//   - Complexity: claimed trace value
//
// Private witness:
//   - BInv: inverse of (K + lambda I)
//   - R   : random vectors for Freivalds checks
type RidgeComplexityCircuit struct {
	K          [N][N]frontend.Variable `gnark:",public"`
	Lambda     frontend.Variable        `gnark:",public"`
	BInv       [N][N]frontend.Variable  // witness for inverse
	Complexity frontend.Variable        `gnark:",public"`
	R          [Trials][N]frontend.Variable
}

func (c *RidgeComplexityCircuit) Define(api frontend.API) error {
	// Freivalds inverse checks: for each random vector r, verify B * (BInv * r) = r
	for t := 0; t < Trials; t++ {
		// u = BInv * r
		var u [N]frontend.Variable
		for i := 0; i < N; i++ {
			sum := frontend.Variable(0)
			for j := 0; j < N; j++ {
				sum = api.Add(sum, api.Mul(c.BInv[i][j], c.R[t][j]))
			}
			u[i] = sum
		}
		// v = B * u  where B = K + lambda I
		var v [N]frontend.Variable
		for i := 0; i < N; i++ {
			sum := frontend.Variable(0)
			for j := 0; j < N; j++ {
				b := c.K[i][j]
				if i == j {
					b = api.Add(b, c.Lambda)
				}
				sum = api.Add(sum, api.Mul(b, u[j]))
			}
			v[i] = sum
		}
		// Assert v == r
		for i := 0; i < N; i++ {
			api.AssertIsEqual(v[i], c.R[t][i])
		}
	}

	// Compute trace(K * BInv)
	trace := frontend.Variable(0)
	for i := 0; i < N; i++ {
		diag := frontend.Variable(0)
		for k := 0; k < N; k++ {
			diag = api.Add(diag, api.Mul(c.K[i][k], c.BInv[k][i]))
		}
		trace = api.Add(trace, diag)
	}

	api.AssertIsEqual(trace, c.Complexity)
	return nil
}

// -------------------------------
// Helper: modular linear algebra
// -------------------------------
// invertMatrix computes the inverse of matrix A modulo the prime field p.
func invertMatrix(A [][]*big.Int, p *big.Int) ([][]*big.Int, error) {
	n := len(A)
	// Build augmented matrix [A | I]
	aug := make([][]*big.Int, n)
	for i := 0; i < n; i++ {
		aug[i] = make([]*big.Int, 2*n)
		for j := 0; j < n; j++ {
			aug[i][j] = new(big.Int).Mod(A[i][j], p)
		}
		for j := 0; j < n; j++ {
			if i == j {
				aug[i][n+j] = big.NewInt(1)
			} else {
				aug[i][n+j] = big.NewInt(0)
			}
		}
	}

	// Gauss-Jordan elimination
	for col := 0; col < n; col++ {
		pivot := -1
		for row := col; row < n; row++ {
			if aug[row][col].Sign() != 0 {
				pivot = row
				break
			}
		}
		if pivot == -1 {
			return nil, fmt.Errorf("matrix not invertible")
		}
		// swap rows
		if pivot != col {
			aug[pivot], aug[col] = aug[col], aug[pivot]
		}

		// normalize pivot row
		inv := new(big.Int).ModInverse(aug[col][col], p)
		for j := 0; j < 2*n; j++ {
			aug[col][j].Mul(aug[col][j], inv)
			aug[col][j].Mod(aug[col][j], p)
		}

		// eliminate other rows
		for row := 0; row < n; row++ {
			if row == col {
				continue
			}
			factor := new(big.Int).Set(aug[row][col])
			if factor.Sign() == 0 {
				continue
			}
			for j := 0; j < 2*n; j++ {
				tmp := new(big.Int).Mul(factor, aug[col][j])
				tmp.Mod(tmp, p)
				aug[row][j].Sub(aug[row][j], tmp)
				aug[row][j].Mod(aug[row][j], p)
			}
		}
	}

	// Extract inverse
	invA := make([][]*big.Int, n)
	for i := 0; i < n; i++ {
		invA[i] = make([]*big.Int, n)
		for j := 0; j < n; j++ {
			invA[i][j] = aug[i][n+j]
		}
	}
	return invA, nil
}

// traceKInv computes trace(K * invB) modulo p.
func traceKInv(K [][]*big.Int, invB [][]*big.Int, p *big.Int) *big.Int {
	res := big.NewInt(0)
	for i := 0; i < len(K); i++ {
		diag := big.NewInt(0)
		for k := 0; k < len(K); k++ {
			tmp := new(big.Int).Mul(K[i][k], invB[k][i])
			tmp.Mod(tmp, p)
			diag.Add(diag, tmp)
			diag.Mod(diag, p)
		}
		res.Add(res, diag)
		res.Mod(res, p)
	}
	return res
}

// -------------------------------
// Example witness construction
// -------------------------------
func main() {
	fmt.Println("Compiling ridge complexity circuit...")
	var circuit RidgeComplexityCircuit
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		panic(err)
	}
	fmt.Println("Number of constraints:", ccs.GetNbConstraints())

	// Build a concrete witness.
	mod := ecc.BN254.ScalarField()

	// Simple positive-definite-ish K: diagonal dominant.
	K := make([][]*big.Int, N)
	for i := 0; i < N; i++ {
		K[i] = make([]*big.Int, N)
		for j := 0; j < N; j++ {
			if i == j {
				K[i][j] = big.NewInt(int64(3 + i))
			} else {
				K[i][j] = big.NewInt(int64(1))
			}
		}
	}

	lambda := big.NewInt(2)

	// Build B = K + lambda * I
	B := make([][]*big.Int, N)
	for i := 0; i < N; i++ {
		B[i] = make([]*big.Int, N)
		for j := 0; j < N; j++ {
			val := new(big.Int).Set(K[i][j])
			if i == j {
				val.Add(val, lambda)
			}
			val.Mod(val, mod)
			B[i][j] = val
		}
	}

	BInv, err := invertMatrix(B, mod)
	if err != nil {
		panic(err)
	}

	complexity := traceKInv(K, BInv, mod)

	// Populate witness
	var wit RidgeComplexityCircuit
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			wit.K[i][j] = new(big.Int).Set(K[i][j])
			wit.BInv[i][j] = new(big.Int).Set(BInv[i][j])
		}
	}
	wit.Lambda = new(big.Int).Set(lambda)
	wit.Complexity = new(big.Int).Set(complexity)
	// Fill Freivalds random vectors deterministically for demo
	for t := 0; t < Trials; t++ {
		for i := 0; i < N; i++ {
			wit.R[t][i] = big.NewInt(int64(3 + t + i))
		}
	}

	prover, err := frontend.NewWitness(&wit, ecc.BN254.ScalarField())
	if err != nil {
		panic(err)
	}
	public, err := prover.Public()
	if err != nil {
		panic(err)
	}

	pk, vk, err := groth16.Setup(ccs)
	if err != nil {
		panic(err)
	}
	proof, err := groth16.Prove(ccs, pk, prover)
	if err != nil {
		fmt.Println("Prove error:", err)
		return
	}
	if err := groth16.Verify(proof, vk, public); err != nil {
		fmt.Println("Verify error:", err)
	} else {
		fmt.Println("Proof verified: true")
	}
}

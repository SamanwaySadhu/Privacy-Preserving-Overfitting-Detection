package main

import (
	"fmt"
	"math/big"

	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

const N = 20

// Trials controls number of Freivalds repetitions for inverse check.
const Trials = 2

type Circuit struct {
	A        [N][N]frontend.Variable `gnark:",public"`
	DetClaim frontend.Variable       `gnark:",public"`
	AInv     [N][N]frontend.Variable // witness inverse of A
	R        [Trials][N]frontend.Variable
}

func (c *Circuit) Define(api frontend.API) error {

	// Determinant via Gaussian elimination (exact, as in original).
	var M [N][N]frontend.Variable
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			M[i][j] = c.A[i][j]
		}
	}

	for k := 0; k < N; k++ {
		pivot := M[k][k]
		inv := api.Inverse(pivot)
		for i := k + 1; i < N; i++ {
			m := api.Mul(M[i][k], inv)
			for j := k; j < N; j++ {
				M[i][j] = api.Sub(M[i][j], api.Mul(m, M[k][j]))
			}
		}
	}

	det := frontend.Variable(1)
	for i := 0; i < N; i++ {
		det = api.Mul(det, M[i][i])
	}

	api.AssertIsEqual(det, c.DetClaim)

	// Freivalds inverse check: A * (AInv * r) == r for random witness r.
	for t := 0; t < Trials; t++ {
		var u [N]frontend.Variable
		for i := 0; i < N; i++ {
			sum := frontend.Variable(0)
			for j := 0; j < N; j++ {
				sum = api.Add(sum, api.Mul(c.AInv[i][j], c.R[t][j]))
			}
			u[i] = sum
		}
		for i := 0; i < N; i++ {
			sum := frontend.Variable(0)
			for j := 0; j < N; j++ {
				sum = api.Add(sum, api.Mul(c.A[i][j], u[j]))
			}
			api.AssertIsEqual(sum, c.R[t][i])
		}
	}
	return nil
}

func buildUpperTriangular20() ([N][N]*big.Int, *big.Int) {
	var A [N][N]*big.Int
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			A[i][j] = new(big.Int)
			if j < i {
				A[i][j].SetInt64(0)
			} else if j == i {
				A[i][j].SetInt64(int64(i + 1))
			} else {
				A[i][j].SetInt64(int64(i + j + 2))
			}
		}
	}
	det := new(big.Int).SetInt64(1)
	for d := 1; d <= N; d++ {
		det.Mul(det, big.NewInt(int64(d)))
	}
	return A, det
}

// invertMatrix computes inverse of A modulo p.
func invertMatrix(A [N][N]*big.Int, p *big.Int) ([N][N]*big.Int, error) {
	// Build augmented matrix [A | I]
	aug := make([][]*big.Int, N)
	for i := 0; i < N; i++ {
		aug[i] = make([]*big.Int, 2*N)
		for j := 0; j < N; j++ {
			aug[i][j] = new(big.Int).Mod(A[i][j], p)
		}
		for j := 0; j < N; j++ {
			if i == j {
				aug[i][N+j] = big.NewInt(1)
			} else {
				aug[i][N+j] = big.NewInt(0)
			}
		}
	}
	// Gauss-Jordan
	for col := 0; col < N; col++ {
		pivot := -1
		for row := col; row < N; row++ {
			if aug[row][col].Sign() != 0 {
				pivot = row
				break
			}
		}
		if pivot == -1 {
			return [N][N]*big.Int{}, fmt.Errorf("matrix not invertible")
		}
		if pivot != col {
			aug[pivot], aug[col] = aug[col], aug[pivot]
		}
		inv := new(big.Int).ModInverse(aug[col][col], p)
		for j := 0; j < 2*N; j++ {
			aug[col][j].Mul(aug[col][j], inv)
			aug[col][j].Mod(aug[col][j], p)
		}
		for row := 0; row < N; row++ {
			if row == col {
				continue
			}
			factor := new(big.Int).Set(aug[row][col])
			if factor.Sign() == 0 {
				continue
			}
			for j := 0; j < 2*N; j++ {
				tmp := new(big.Int).Mul(factor, aug[col][j])
				tmp.Mod(tmp, p)
				aug[row][j].Sub(aug[row][j], tmp)
				aug[row][j].Mod(aug[row][j], p)
			}
		}
	}
	var invA [N][N]*big.Int
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			invA[i][j] = aug[i][N+j]
		}
	}
	return invA, nil
}

func main() {
	var circuit Circuit
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		panic(err)
	}

	Avals, det := buildUpperTriangular20()

	var wit Circuit
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			wit.A[i][j] = new(big.Int).Set(Avals[i][j])
		}
	}
	wit.DetClaim = new(big.Int).Set(det)

	// Build inverse of A for Freivalds witness
	invA, err := invertMatrix(Avals, ecc.BN254.ScalarField())
	if err != nil {
		panic(err)
	}
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			wit.AInv[i][j] = new(big.Int).Set(invA[i][j])
		}
	}
	// Deterministic random vectors for demo
	for t := 0; t < Trials; t++ {
		for i := 0; i < N; i++ {
			wit.R[t][i] = big.NewInt(int64(5 + t + i))
		}
	}

	proverWitness, _ := frontend.NewWitness(&wit, ecc.BN254.ScalarField())
	publicWitness, _ := proverWitness.Public()

	pk, vk, _ := groth16.Setup(ccs)
	proof, _ := groth16.Prove(ccs, pk, proverWitness)
	err = groth16.Verify(proof, vk, publicWitness)
	fmt.Println("Determinant proof verified:", err == nil)
}

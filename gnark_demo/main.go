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

// Circuit: proves det(A) == DetClaim using Gaussian elimination (no row swaps)
type Circuit struct {
	A        [N][N]frontend.Variable `gnark:",public"`
	DetClaim frontend.Variable        `gnark:",public"`
}

func (c *Circuit) Define(api frontend.API) error {

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

	proverWitness, _ := frontend.NewWitness(&wit, ecc.BN254.ScalarField())
	publicWitness, _ := proverWitness.Public()

	pk, vk, _ := groth16.Setup(ccs)
	proof, _ := groth16.Prove(ccs, pk, proverWitness)
	err = groth16.Verify(proof, vk, publicWitness)
	fmt.Println("Determinant proof verified:", err == nil)
}

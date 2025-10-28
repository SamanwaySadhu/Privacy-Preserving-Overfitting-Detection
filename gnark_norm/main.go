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
const (
	NumMatrices = 18
	Rows        = 224
	Cols        = 224
)

// Circuit verifying the remaining "spectral_complexity" computation:
//
// For each matrix W_i:
//   1) Form (W_i^T - I).
//   2) For each column j, compute sumSquares_j = Σ_r (diff[r][j])^2.
//      Verify a provided witness root_j satisfies root_j^2 = sumSquares_j.
//   3) Sum roots: s_i = Σ_j root_j.
//   4) Verify 2/3 power via t_i^3 = s_i^2, where t_i is a witness.
//   5) Accumulate Total = Σ_i t_i; Total is public.
type SpectralComplexityCircuit struct {
	W        [NumMatrices][Rows][Cols]frontend.Variable `gnark:",public"` // matrices 
	Roots    [NumMatrices][Cols]frontend.Variable       // column L2 norms (sqrt witnesses)
	TwoThird [NumMatrices]frontend.Variable             // s_i^(2/3) witnesses (t_i)
	Total    frontend.Variable                          `gnark:",public"` // Σ_i t_i
}

func (c *SpectralComplexityCircuit) Define(api frontend.API) error {
	total := frontend.Variable(0)

	for i := 0; i < NumMatrices; i++ {
		// Build column-wise norms of (W^T - I)
		sumCols := frontend.Variable(0)
		for j := 0; j < Cols; j++ {
			sumSquares := frontend.Variable(0)
			for r := 0; r < Rows; r++ {
				// diff = (W^T - I)[r, j] = W[r][j] - (r==j ? 1 : 0)
				val := c.W[i][r][j]
				if r == j {
					val = api.Sub(val, 1)
				}
				sumSquares = api.Add(sumSquares, api.Mul(val, val))
			}
			root := c.Roots[i][j]
			// root^2 == sumSquares
			api.AssertIsEqual(api.Mul(root, root), sumSquares)
			sumCols = api.Add(sumCols, root)
		}

		// Verify 2/3-power consistency: t^3 == (sumCols)^2
		t := c.TwoThird[i]
		api.AssertIsEqual(api.Mul(t, t, t), api.Mul(sumCols, sumCols))
		total = api.Add(total, t)
	}

	api.AssertIsEqual(total, c.Total)
	return nil
}

// -------------------------------
// Example witness construction
// -------------------------------
//
// To guarantee satisfiable constraints, we *construct* W so that:
//  • (W^T - I) has non-zero entries only on the diagonal, equal to chosen roots r_j.
//    This makes sumSquares_j = r_j^2 and root_j = r_j consistent.
//  • Let s_i = Σ_j r_j. We choose s_i to be a perfect cube: s_i = u_i^3.
//    Then set t_i = u_i^2 so that t_i^3 = s_i^2 holds identically.
//
// Concretely, we set for each matrix i and column j:
//    r_0 = u_i^3, and r_1 = r_2 = ... = r_{Cols-1} = 0.
//    Then W[i][j][j] = 1 + r_j, all off-diagonals = 0.
//    => sumSquares_0 = (u_i^3)^2, root_0 = u_i^3, s_i = u_i^3, t_i = u_i^2, OK.

func main() {
	fmt.Println("Compiling circuit...")
	var circuit SpectralComplexityCircuit
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		panic(err)
	}
	fmt.Println("Number of constraints:", ccs.GetNbConstraints())

	// Build satisfying witness
	var wit SpectralComplexityCircuit
	total := big.NewInt(0)

	for i := 0; i < NumMatrices; i++ {
		// Choose u_i (small positive integer)
		u := big.NewInt(int64(2 + i)) // 2, 3, 4, ...
		u2 := new(big.Int).Mul(u, u)  // u^2
		u3 := new(big.Int).Mul(u2, u) // u^3

		// Roots r_j: r_0 = u^3 ; others = 0
		for j := 0; j < Cols; j++ {
			if j == 0 {
				wit.Roots[i][j] = new(big.Int).Set(u3)
			} else {
				wit.Roots[i][j] = big.NewInt(0)
			}
		}

		// Construct W so that (W^T - I) column j equals root r_j on the diagonal only:
		// W[r][j] = 0 for r != j; W[j][j] = 1 + r_j
		for r := 0; r < Rows; r++ {
			for c := 0; c < Cols; c++ {
				wit.W[i][r][c] = big.NewInt(0)
			}
		}
		for j := 0; j < Cols; j++ {
			diag := new(big.Int)
			diag.Add(big.NewInt(1), wit.Roots[i][j].(*big.Int)) // 1 + r_j
			// place on diagonal entry W[j][j]
			if j < Rows { // guard if Rows != Cols
				wit.W[i][j][j] = diag
			}
		}

		// s_i = Σ_j r_j = u^3
		// pick t_i = u^2 so that t_i^3 = (u^2)^3 = u^6 = (u^3)^2 = s_i^2
		wit.TwoThird[i] = new(big.Int).Set(u2)
		total.Add(total, u2)
	}

	wit.Total = new(big.Int).Set(total)

	// Prove & Verify (with error checks to avoid nil-proof panic)
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

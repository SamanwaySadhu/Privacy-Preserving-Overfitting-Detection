package main

import (
	"fmt"
	"math/big"

	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark-crypto/ecc"
	fr "github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

const N = 38 // 38*38 matrix

// Circuit: prove det(A) == DetClaim using Gaussian elimination with
// single-step pivoting (swap with next row if pivot==0), sign tracking,
// and safe constraints that never require taking an inverse inside the circuit.
//
// We introduce secret witness variables Mult[k][i] (for i>k) that represent
// elimination multipliers. The circuit enforces:
//   nz * M[i][k] == pivot * Mult[k][i]
// where nz = 1 - IsZero(pivot). When pivot == 0, nz == 0 and this reduces to 0 == 0,
// avoiding inverse-on-zero. Row updates then use Mult[k][i].
type Circuit struct {
	A        [N][N]frontend.Variable `gnark:",public"` // public matrix
	DetClaim frontend.Variable        `gnark:",public"` // public claimed determinant

	// Secret witness multipliers for elimination (only entries with i>k are used).
	Mult [N][N]frontend.Variable
}

func (c *Circuit) Define(api frontend.API) error {
	// Copy matrix A -> M (symbolic variables)
	var M [N][N]frontend.Variable
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			M[i][j] = c.A[i][j]
		}
	}

	sign := frontend.Variable(1) // starts at +1

	for k := 0; k < N; k++ {
		// Check pivot; if zero and k < N-1, swap with next row and flip sign
		pivot0 := M[k][k]
		isZero0 := api.IsZero(pivot0) // 1 if zero else 0

		if k < N-1 {
			// Row swap with (k+1) if pivot0 == 0
			for j := 0; j < N; j++ {
				// swap(M[k][j], M[k+1][j]) controlled by isZero0
				a := M[k][j]
				b := M[k+1][j]
				M[k][j] = api.Select(isZero0, b, a)
				M[k+1][j] = api.Select(isZero0, a, b)
			}
			// sign *= (isZero0 ? -1 : 1)
			minusOne := api.Sub(0, 1)                       // -1 in the field
			swapSign := api.Add(api.Mul(minusOne, isZero0), // -1 * 1 = -1 when swapping
				api.Sub(1, isZero0)) // + (1 - 1) = 0  → total = -1 when swap
			sign = api.Mul(sign, swapSign)
		}

		// Re-read pivot after possible swap
		pivot := M[k][k]
		isZero := api.IsZero(pivot)
		nz := api.Sub(1, isZero) // 1 if pivot != 0 else 0

		// For each row i>k, enforce nz * M[i][k] == pivot * Mult[k][i]
		// and update the row: M[i][j] -= Mult[k][i] * M[k][j]
		for i := k + 1; i < N; i++ {
			m := c.Mult[k][i] // secret witness multiplier

			// Constraint: nz * M[i][k] == pivot * m
			api.AssertIsEqual(api.Mul(nz, M[i][k]), api.Mul(pivot, m))

			for j := k; j < N; j++ {
				M[i][j] = api.Sub(M[i][j], api.Mul(m, M[k][j]))
			}
		}
	}

	// det = sign * product(diagonal)
	det := frontend.Variable(1)
	for i := 0; i < N; i++ {
		det = api.Mul(det, M[i][i])
	}
	det = api.Mul(det, sign)

	api.AssertIsEqual(det, c.DetClaim)
	return nil
}

//
// ----------------------
// Native witness builder
// ----------------------
//
// Everything below runs in Go (off-circuit) to construct a *consistent* witness.
// We do elimination *mod the BN254 field prime* so divisions are modular inverses,
// and we mirror the same "swap-with-next-if-zero" pivot rule.
//

// mod helpers
var modulus *big.Int = fr.Modulus() // BN254 field modulus as *big.Int


func mod(x *big.Int) *big.Int {
	r := new(big.Int).Mod(x, modulus)
	if r.Sign() < 0 {
		r.Add(r, modulus)
	}
	return r
}

func addMod(a, b *big.Int) *big.Int {
	t := new(big.Int).Add(a, b)
	return mod(t)
}
func subMod(a, b *big.Int) *big.Int {
	t := new(big.Int).Sub(a, b)
	return mod(t)
}
func mulMod(a, b *big.Int) *big.Int {
	t := new(big.Int).Mul(a, b)
	return mod(t)
}
func invMod(a *big.Int) *big.Int {
	if a.Sign() == 0 {
		return new(big.Int) // 0 (we won't use it when a==0)
	}
	// inverse via Exp: a^(p-2) mod p
	return new(big.Int).ModInverse(a, modulus)
}

// buildWitness performs the same elimination as the circuit:
// - if pivot is 0 and k < N-1, swap with next row and flip sign
// - if pivot != 0, for each i>k, m = M[i][k] * inv(pivot) mod p
//   then Row_i -= m * Row_k
// It returns (A_pub, Det_pub, Mult_witness).
func buildWitness(Ain [N][N]*big.Int) (Aout [N][N]*big.Int, det *big.Int, Mult [N][N]*big.Int) {
	// Copy A, reduce mod p
	M := make([][]*big.Int, N)
	for i := 0; i < N; i++ {
		M[i] = make([]*big.Int, N)
		for j := 0; j < N; j++ {
			M[i][j] = mod(new(big.Int).Set(Ain[i][j]))
			Aout[i][j] = new(big.Int).Set(M[i][j]) // public A entries are mod p
		}
	}

	sign := big.NewInt(1)        // +1
	minusOne := new(big.Int).Sub(modulus, big.NewInt(1)) // -1 mod p

	// elimination
	for k := 0; k < N; k++ {
		// if pivot is zero and k < N-1, swap with next row
		if M[k][k].Sign() == 0 && k < N-1 {
			M[k], M[k+1] = M[k+1], M[k]
			sign = mulMod(sign, minusOne) // flip sign
		}

		pivot := M[k][k]
		if pivot.Sign() != 0 {
			inv := invMod(pivot)
			for i := k + 1; i < N; i++ {
				m := mulMod(M[i][k], inv)
				Mult[k][i] = m
				for j := k; j < N; j++ {
					M[i][j] = subMod(M[i][j], mulMod(m, M[k][j]))
				}
			}
		} else {
			// pivot == 0 → set multipliers to 0; no row update
			for i := k + 1; i < N; i++ {
				Mult[k][i] = new(big.Int)
			}
		}
	}

	// det = sign * product(diagonal) mod p
	d := new(big.Int).Set(sign)
	for i := 0; i < N; i++ {
		d = mulMod(d, M[i][i])
	}
	return Aout, d, Mult
}

// Example: build a dense-ish integer matrix (mod p) that is usually nonsingular.
func buildExampleMatrix() [N][N]*big.Int {
	var A [N][N]*big.Int
	// simple pattern with non-trivial structure
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			// small integers, then mod p in witness builder
			val := int64((3*i + 7*j + 11)%(1<<16) + 1) // avoid zeros
			A[i][j] = big.NewInt(val)
		}
	}
	return A
}

func main() {
	// 1) Compile circuit
	fmt.Println("compiling circuit ...")
	var circuit Circuit
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		panic(err)
	}
	fmt.Printf("nbConstraints=%d\n", ccs.GetNbConstraints())

	// 2) Build example matrix and a consistent witness (A public, Mult secret, Det public)
	Araw := buildExampleMatrix()
	Apub, det, Mult := buildWitness(Araw)
	fmt.Println("Public determinant (mod BN254):", det)

	// 3) Fill witness
	var wit Circuit
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			wit.A[i][j] = new(big.Int).Set(Apub[i][j])
			wit.Mult[i][j] = new(big.Int) // initialize; many entries unused
		}
	}
	// place only used multipliers (i>k)
	for k := 0; k < N; k++ {
		for i := k + 1; i < N; i++ {
			if Mult[k][i] == nil {
				wit.Mult[k][i] = new(big.Int) // 0
			} else {
				wit.Mult[k][i] = new(big.Int).Set(Mult[k][i])
			}
		}
	}
	wit.DetClaim = new(big.Int).Set(det)

	// 4) Prover & public witnesses
	proverWitness, err := frontend.NewWitness(&wit, ecc.BN254.ScalarField())
	if err != nil {
		panic(err)
	}
	publicWitness, err := proverWitness.Public()
	if err != nil {
		panic(err)
	}

	// 5) Setup / Prove / Verify
	pk, vk, err := groth16.Setup(ccs)
	if err != nil {
		panic(err)
	}
	proof, err := groth16.Prove(ccs, pk, proverWitness)
	if err != nil {
		// show helpful error instead of nil proof panic
		fmt.Println("Prove error:", err)
		return
	}
	if err := groth16.Verify(proof, vk, publicWitness); err != nil {
		fmt.Println("Verify error:", err)
	} else {
		fmt.Println("Proof verified: true")
	}
}

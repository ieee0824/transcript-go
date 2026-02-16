#include "textflag.h"

// func mahalanobisAsm(x, mean, invVar *float64, n int) float64
//
// Computes sum((x[i]-mean[i])^2 * invVar[i]) using AMD64 SSE2.
// Processes 2 float64 per iteration (128-bit XMM registers).
TEXT ·mahalanobisAsm(SB), NOSPLIT, $0-40
	MOVQ	x+0(FP), AX		// x pointer
	MOVQ	mean+8(FP), BX		// mean pointer
	MOVQ	invVar+16(FP), CX	// invVar pointer
	MOVQ	n+24(FP), DX		// length

	// Zero accumulator X0
	XORPD	X0, X0

	// Main loop: process 2 float64 per iteration
	CMPQ	DX, $2
	JL	tail

loop2:
	MOVUPD	(AX), X1		// X1 = {x[i], x[i+1]}
	MOVUPD	(BX), X2		// X2 = {mean[i], mean[i+1]}
	SUBPD	X2, X1			// X1 = x - mean
	MULPD	X1, X1			// X1 = diff^2
	MOVUPD	(CX), X3		// X3 = {invVar[i], invVar[i+1]}
	MULPD	X3, X1			// X1 = diff^2 * invVar
	ADDPD	X1, X0			// X0 += result

	ADDQ	$16, AX
	ADDQ	$16, BX
	ADDQ	$16, CX
	SUBQ	$2, DX
	CMPQ	DX, $2
	JGE	loop2

tail:
	// Horizontal reduction: X0[low] + X0[high]
	MOVHLPS	X0, X1			// X1[low] = X0[high]
	ADDSD	X1, X0			// X0[low] += X1[low]

	// Handle remaining element (if n was odd)
	CMPQ	DX, $0
	JE	done

	MOVSD	(AX), X1		// x[i]
	MOVSD	(BX), X2		// mean[i]
	SUBSD	X2, X1			// diff = x - mean
	MULSD	X1, X1			// diff^2
	MOVSD	(CX), X3		// invVar[i]
	MULSD	X3, X1			// diff^2 * invVar
	ADDSD	X1, X0			// sum += result

done:
	MOVSD	X0, ret+32(FP)
	RET

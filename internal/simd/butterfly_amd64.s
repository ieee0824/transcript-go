#include "textflag.h"

// func butterflyAsm(uRe, uIm, vRe, vIm, twRe, twIm *float64, n int)
//
// FFT butterfly on split R/I arrays using AMD64 SSE2.
// Processes 2 float64 per iteration (128-bit XMM registers).
TEXT ·butterflyAsm(SB), NOSPLIT, $0-56
	MOVQ	uRe+0(FP), AX		// uRe pointer
	MOVQ	uIm+8(FP), BX		// uIm pointer
	MOVQ	vRe+16(FP), CX		// vRe pointer
	MOVQ	vIm+24(FP), DX		// vIm pointer
	MOVQ	twRe+32(FP), SI		// twRe pointer
	MOVQ	twIm+40(FP), DI		// twIm pointer
	MOVQ	n+48(FP), R8		// count

	// Main loop: process 2 butterflies per iteration
	CMPQ	R8, $2
	JL	tail

loop2:
	MOVUPD	(AX), X0		// X0 = {uRe[k], uRe[k+1]}
	MOVUPD	(BX), X1		// X1 = {uIm[k], uIm[k+1]}
	MOVUPD	(CX), X2		// X2 = {vRe[k], vRe[k+1]}
	MOVUPD	(DX), X3		// X3 = {vIm[k], vIm[k+1]}
	MOVUPD	(SI), X4		// X4 = {twRe[k], twRe[k+1]}
	MOVUPD	(DI), X5		// X5 = {twIm[k], twIm[k+1]}

	// t_re = twRe*vRe - twIm*vIm
	MOVAPD	X4, X6			// X6 = twRe
	MULPD	X2, X6			// X6 = twRe * vRe
	MOVAPD	X5, X7			// X7 = twIm
	MULPD	X3, X7			// X7 = twIm * vIm
	SUBPD	X7, X6			// X6 = t_re

	// t_im = twRe*vIm + twIm*vRe
	MOVAPD	X4, X7			// X7 = twRe
	MULPD	X3, X7			// X7 = twRe * vIm
	MOVAPD	X5, X8			// X8 = twIm
	MULPD	X2, X8			// X8 = twIm * vRe
	ADDPD	X8, X7			// X7 = t_im

	// u' = u + t, v' = u - t
	MOVAPD	X0, X8			// X8 = save uRe
	ADDPD	X6, X0			// X0 = uRe + t_re (new uRe)
	SUBPD	X6, X8			// X8 = uRe - t_re (new vRe)
	MOVAPD	X1, X9			// X9 = save uIm
	ADDPD	X7, X1			// X1 = uIm + t_im (new uIm)
	SUBPD	X7, X9			// X9 = uIm - t_im (new vIm)

	// Store results
	MOVUPD	X0, (AX)
	MOVUPD	X1, (BX)
	MOVUPD	X8, (CX)
	MOVUPD	X9, (DX)

	// Advance pointers
	ADDQ	$16, AX
	ADDQ	$16, BX
	ADDQ	$16, CX
	ADDQ	$16, DX
	ADDQ	$16, SI
	ADDQ	$16, DI
	SUBQ	$2, R8
	CMPQ	R8, $2
	JGE	loop2

tail:
	// Handle remaining element (if n was odd)
	CMPQ	R8, $0
	JE	done

	// Scalar butterfly for 1 element
	MOVSD	(CX), X2		// vRe
	MOVSD	(DX), X3		// vIm
	MOVSD	(SI), X4		// twRe
	MOVSD	(DI), X5		// twIm
	MOVSD	(AX), X0		// uRe
	MOVSD	(BX), X1		// uIm

	// t_re = twRe*vRe - twIm*vIm
	MOVSD	X4, X6
	MULSD	X2, X6			// twRe*vRe
	MOVSD	X5, X7
	MULSD	X3, X7			// twIm*vIm
	SUBSD	X7, X6			// t_re

	// t_im = twRe*vIm + twIm*vRe
	MOVSD	X4, X7
	MULSD	X3, X7			// twRe*vIm
	MOVSD	X5, X8
	MULSD	X2, X8			// twIm*vRe
	ADDSD	X8, X7			// t_im

	// u' = u + t, v' = u - t
	MOVSD	X0, X8
	ADDSD	X6, X0			// new uRe
	SUBSD	X6, X8			// new vRe
	MOVSD	X1, X9
	ADDSD	X7, X1			// new uIm
	SUBSD	X7, X9			// new vIm

	MOVSD	X0, (AX)
	MOVSD	X1, (BX)
	MOVSD	X8, (CX)
	MOVSD	X9, (DX)

done:
	RET

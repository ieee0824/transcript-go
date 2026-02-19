#include "textflag.h"

// func butterflyAsm(uRe, uIm, vRe, vIm, twRe, twIm *float64, n int)
//
// FFT butterfly on split R/I arrays using ARM64 NEON.
// Processes 2 float64 per iteration (128-bit NEON registers).
//
// For each k:
//   t_re = twRe[k]*vRe[k] - twIm[k]*vIm[k]
//   t_im = twRe[k]*vIm[k] + twIm[k]*vRe[k]
//   uRe[k], vRe[k] = uRe[k]+t_re, uRe[k]-t_re
//   uIm[k], vIm[k] = uIm[k]+t_im, uIm[k]-t_im
TEXT ·butterflyAsm(SB), NOSPLIT, $0-56
	MOVD	uRe+0(FP), R0		// uRe pointer
	MOVD	uIm+8(FP), R1		// uIm pointer
	MOVD	vRe+16(FP), R2		// vRe pointer
	MOVD	vIm+24(FP), R3		// vIm pointer
	MOVD	twRe+32(FP), R4		// twRe pointer
	MOVD	twIm+40(FP), R5		// twIm pointer
	MOVD	n+48(FP), R6		// count

	// Main loop: process 2 butterflies per iteration
	CMP	$2, R6
	BLT	tail

loop2:
	VLD1	(R0), [V0.D2]		// V0 = {uRe[k], uRe[k+1]}
	VLD1	(R1), [V1.D2]		// V1 = {uIm[k], uIm[k+1]}
	VLD1	(R2), [V2.D2]		// V2 = {vRe[k], vRe[k+1]}
	VLD1	(R3), [V3.D2]		// V3 = {vIm[k], vIm[k+1]}
	VLD1	(R4), [V4.D2]		// V4 = {twRe[k], twRe[k+1]}
	VLD1	(R5), [V5.D2]		// V5 = {twIm[k], twIm[k+1]}

	// t_re = twRe*vRe - twIm*vIm
	// FMUL V6.2D, V4.2D, V2.2D  (V6 = twRe * vRe)
	WORD	$0x6E62DC86
	// FMLS V6.2D, V5.2D, V3.2D  (V6 -= twIm * vIm → t_re)
	WORD	$0x4EE3CCA6

	// t_im = twRe*vIm + twIm*vRe
	// FMUL V7.2D, V4.2D, V3.2D  (V7 = twRe * vIm)
	WORD	$0x6E63DC87
	// FMLA V7.2D, V5.2D, V2.2D  (V7 += twIm * vRe → t_im)
	WORD	$0x4E62CCA7

	// u' = u + t
	// FADD V8.2D, V0.2D, V6.2D  (new uRe)
	WORD	$0x4E66D408
	// FADD V9.2D, V1.2D, V7.2D  (new uIm)
	WORD	$0x4E67D429

	// v' = u - t
	// FSUB V10.2D, V0.2D, V6.2D  (new vRe)
	WORD	$0x4EE6D40A
	// FSUB V11.2D, V1.2D, V7.2D  (new vIm)
	WORD	$0x4EE7D42B

	// Store results
	VST1	[V8.D2], (R0)
	VST1	[V9.D2], (R1)
	VST1	[V10.D2], (R2)
	VST1	[V11.D2], (R3)

	// Advance pointers
	ADD	$16, R0
	ADD	$16, R1
	ADD	$16, R2
	ADD	$16, R3
	ADD	$16, R4
	ADD	$16, R5
	SUB	$2, R6
	CMP	$2, R6
	BGE	loop2

tail:
	// Handle remaining element (if n was odd)
	CBZ	R6, done

	// Load all values
	FMOVD	(R2), F2		// vRe
	FMOVD	(R3), F3		// vIm
	FMOVD	(R4), F4		// twRe
	FMOVD	(R5), F5		// twIm
	FMOVD	(R0), F0		// uRe
	FMOVD	(R1), F1		// uIm

	// t_re = twRe*vRe - twIm*vIm
	FMULD	F2, F4, F6		// F6 = twRe * vRe
	FMULD	F3, F5, F7		// F7 = twIm * vIm
	FSUBD	F7, F6, F6		// F6 = t_re

	// t_im = twRe*vIm + twIm*vRe
	FMULD	F3, F4, F7		// F7 = twRe * vIm
	FMULD	F2, F5, F8		// F8 = twIm * vRe
	FADDD	F8, F7, F7		// F7 = t_im

	// u' = u + t
	FADDD	F6, F0, F8		// F8 = uRe + t_re
	FADDD	F7, F1, F9		// F9 = uIm + t_im

	// v' = u - t
	FSUBD	F6, F0, F10		// F10 = uRe - t_re
	FSUBD	F7, F1, F11		// F11 = uIm - t_im

	// Store
	FMOVD	F8, (R0)
	FMOVD	F9, (R1)
	FMOVD	F10, (R2)
	FMOVD	F11, (R3)

done:
	RET

#include "textflag.h"

// func mahalanobisAsm(x, mean, invVar *float64, n int) float64
//
// Computes sum((x[i]-mean[i])^2 * invVar[i]) using ARM64 NEON.
// Processes 2 float64 per iteration (128-bit NEON registers).
//
// Note: Go assembler does not support VFSUB/VFMUL/VFADD with .D2 (float64 vector).
// We use WORD encoding to emit the raw instruction bytes.
TEXT ·mahalanobisAsm(SB), NOSPLIT, $0-40
	MOVD	x+0(FP), R0		// x pointer
	MOVD	mean+8(FP), R1		// mean pointer
	MOVD	invVar+16(FP), R2	// invVar pointer
	MOVD	n+24(FP), R3		// length

	// Zero accumulator V0
	VEOR	V0.B16, V0.B16, V0.B16

	// Main loop: process 2 float64 per iteration
	CMP	$2, R3
	BLT	tail

loop2:
	VLD1	(R0), [V1.D2]		// V1 = {x[i], x[i+1]}
	VLD1	(R1), [V2.D2]		// V2 = {mean[i], mean[i+1]}
	VLD1	(R2), [V3.D2]		// V3 = {invVar[i], invVar[i+1]}

	// FSUB V4.2D, V1.2D, V2.2D  (V4 = V1 - V2)
	WORD	$0x4EE2D424
	// FMUL V5.2D, V4.2D, V4.2D  (V5 = V4 * V4 = diff^2)
	WORD	$0x6E64DC85
	// FMUL V5.2D, V5.2D, V3.2D  (V5 = V5 * V3 = diff^2 * invVar)
	WORD	$0x6E63DCA5
	// FADD V0.2D, V0.2D, V5.2D  (V0 += V5)
	WORD	$0x4E65D400

	ADD	$16, R0
	ADD	$16, R1
	ADD	$16, R2
	SUB	$2, R3
	CMP	$2, R3
	BGE	loop2

tail:
	// Horizontal reduction: F0 = V0.D[0] + V0.D[1]
	VEXT	$8, V0.B16, V0.B16, V1.B16
	FADDD	F1, F0, F0

	// Handle remaining element (if n was odd)
	CBZ	R3, done

	FMOVD	(R0), F1		// x[i]
	FMOVD	(R1), F2		// mean[i]
	FMOVD	(R2), F3		// invVar[i]
	FSUBD	F2, F1, F1		// diff = x - mean
	FMULD	F1, F1, F1		// diff^2
	FMULD	F3, F1, F1		// diff^2 * invVar
	FADDD	F1, F0, F0		// sum += result

done:
	FMOVD	F0, ret+32(FP)
	RET

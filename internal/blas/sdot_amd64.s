#include "textflag.h"

// func dotAVX2f32(a, b *float32, n int) float32
TEXT ·dotAVX2f32(SB), NOSPLIT, $0-28
	MOVQ	a+0(FP), SI
	MOVQ	b+8(FP), DI
	MOVQ	n+16(FP), CX

	VXORPS	Y0, Y0, Y0
	VXORPS	Y1, Y1, Y1

	CMPQ	CX, $16
	JL	tail8

loop16:
	VMOVUPS	(SI), Y2
	VMOVUPS	32(SI), Y4
	VMOVUPS	(DI), Y3
	VMOVUPS	32(DI), Y5
	VFMADD231PS	Y3, Y2, Y0
	VFMADD231PS	Y5, Y4, Y1
	ADDQ	$64, SI
	ADDQ	$64, DI
	SUBQ	$16, CX
	CMPQ	CX, $16
	JGE	loop16

	VADDPS	Y1, Y0, Y0

tail8:
	CMPQ	CX, $8
	JL	reduce
	VMOVUPS	(SI), Y2
	VMOVUPS	(DI), Y3
	VFMADD231PS	Y3, Y2, Y0
	ADDQ	$32, SI
	ADDQ	$32, DI
	SUBQ	$8, CX

reduce:
	// Horizontal sum of 8 float32 in Y0
	VEXTRACTF128	$1, Y0, X1
	VADDPS	X1, X0, X0
	// X0 = [a, b, c, d]
	VPERMILPS	$0x0E, X0, X1	// X1 = [c, d, ?, ?]
	VADDPS	X1, X0, X0		// X0 = [a+c, b+d, ?, ?]
	VPERMILPS	$0x01, X0, X1	// X1 = [b+d, ?, ?, ?]
	VADDSS	X1, X0, X0		// X0 = [a+b+c+d]

	CMPQ	CX, $0
	JE	done

scalar:
	MOVSS	(SI), X2
	MOVSS	(DI), X3
	VFMADD231SS	X3, X2, X0
	ADDQ	$4, SI
	ADDQ	$4, DI
	DECQ	CX
	JNZ	scalar

done:
	VZEROUPPER
	MOVSS	X0, ret+24(FP)
	RET

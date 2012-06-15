
#line 1 "r.rl"
/*
 *	A ragel lexer for R.
 *	In ragel, the lexer drives the parsing process, so this also has the basic parsing functions.
 *	Use this to generate parser.cpp
 *      TODO: 
 *            Include the double-to-int warnings, e.g. on 1.0L  and 1.5L
 *            Generate hex numbers
 *            Do we really want to allow '.' inside hex numbers? 
 *			R allows them, but ignores them when parsing to a number.
 *            R parser has a rule for OP % OP. Is this ever used?
 */

#include "parser.h"
#include "r.h"
#include "r.cpp"


#line 21 "parser.cpp"
static const char _Scanner_actions[] = {
	0, 1, 2, 1, 3, 1, 25, 1, 
	26, 1, 27, 1, 28, 1, 29, 1, 
	30, 1, 31, 1, 32, 1, 33, 1, 
	34, 1, 35, 1, 36, 1, 37, 1, 
	38, 1, 39, 1, 40, 1, 41, 1, 
	42, 1, 43, 1, 44, 1, 45, 1, 
	46, 1, 47, 1, 48, 1, 49, 1, 
	50, 1, 51, 1, 52, 1, 53, 1, 
	54, 1, 55, 1, 56, 1, 57, 1, 
	58, 1, 59, 1, 60, 1, 61, 1, 
	62, 1, 63, 1, 64, 1, 65, 1, 
	66, 1, 67, 1, 68, 1, 69, 1, 
	70, 1, 71, 1, 72, 1, 73, 1, 
	74, 1, 75, 1, 76, 1, 77, 2, 
	0, 1, 2, 3, 4, 2, 3, 5, 
	2, 3, 6, 2, 3, 7, 2, 3, 
	8, 2, 3, 9, 2, 3, 10, 2, 
	3, 11, 2, 3, 12, 2, 3, 13, 
	2, 3, 14, 2, 3, 15, 2, 3, 
	16, 2, 3, 17, 2, 3, 18, 2, 
	3, 19, 2, 3, 20, 2, 3, 21, 
	2, 3, 22, 2, 3, 23, 2, 3, 
	24
};

static const short _Scanner_key_offsets[] = {
	0, 0, 4, 5, 8, 8, 10, 13, 
	13, 17, 19, 20, 23, 23, 71, 75, 
	79, 80, 81, 82, 83, 84, 92, 100, 
	108, 114, 118, 125, 126, 127, 130, 131, 
	132, 141, 150, 159, 168, 177, 186, 197, 
	205, 216, 225, 234, 243, 252, 261, 270, 
	279, 288, 296, 305, 314, 323, 332, 341, 
	350, 358, 367, 376, 385, 393, 402, 411, 
	420, 429, 438, 447, 448, 457, 466, 475, 
	484, 493, 502, 511, 521, 530, 539, 548, 
	557, 566, 575, 584, 594, 603, 612, 621, 
	630, 639, 648, 657, 666, 675, 684, 693, 
	702
};

static const char _Scanner_trans_keys[] = {
	10, 35, 33, 126, 10, 10, 34, 92, 
	10, 37, 10, 39, 92, 43, 45, 48, 
	57, 48, 57, 45, 10, 92, 96, 10, 
	33, 34, 35, 36, 37, 38, 39, 40, 
	41, 42, 43, 44, 45, 46, 47, 58, 
	59, 60, 61, 62, 63, 64, 70, 73, 
	78, 84, 91, 92, 93, 94, 95, 96, 
	98, 101, 102, 105, 110, 114, 119, 123, 
	124, 125, 126, 48, 57, 65, 122, 10, 
	35, 33, 126, 10, 35, 33, 126, 61, 
	38, 42, 62, 62, 46, 95, 48, 57, 
	65, 90, 97, 122, 46, 95, 48, 57, 
	65, 90, 97, 122, 46, 95, 48, 57, 
	65, 90, 97, 122, 69, 76, 101, 105, 
	48, 57, 76, 105, 48, 57, 46, 69, 
	76, 101, 105, 48, 57, 58, 58, 45, 
	60, 61, 61, 61, 46, 65, 95, 48, 
	57, 66, 90, 97, 122, 46, 76, 95, 
	48, 57, 65, 90, 97, 122, 46, 83, 
	95, 48, 57, 65, 90, 97, 122, 46, 
	69, 95, 48, 57, 65, 90, 97, 122, 
	46, 95, 110, 48, 57, 65, 90, 97, 
	122, 46, 95, 102, 48, 57, 65, 90, 
	97, 122, 46, 65, 85, 95, 97, 48, 
	57, 66, 90, 98, 122, 46, 95, 48, 
	57, 65, 90, 97, 122, 46, 95, 99, 
	105, 114, 48, 57, 65, 90, 97, 122, 
	46, 95, 104, 48, 57, 65, 90, 97, 
	122, 46, 95, 97, 48, 57, 65, 90, 
	98, 122, 46, 95, 114, 48, 57, 65, 
	90, 97, 122, 46, 95, 97, 48, 57, 
	65, 90, 98, 122, 46, 95, 99, 48, 
	57, 65, 90, 97, 122, 46, 95, 116, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	101, 48, 57, 65, 90, 97, 122, 46, 
	95, 114, 48, 57, 65, 90, 97, 122, 
	46, 95, 48, 57, 65, 90, 97, 122, 
	46, 95, 110, 48, 57, 65, 90, 97, 
	122, 46, 95, 116, 48, 57, 65, 90, 
	97, 122, 46, 95, 101, 48, 57, 65, 
	90, 97, 122, 46, 95, 103, 48, 57, 
	65, 90, 97, 122, 46, 95, 101, 48, 
	57, 65, 90, 97, 122, 46, 95, 114, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	101, 48, 57, 65, 90, 97, 122, 46, 
	95, 97, 48, 57, 65, 90, 98, 122, 
	46, 95, 108, 48, 57, 65, 90, 97, 
	122, 46, 95, 48, 57, 65, 90, 97, 
	122, 46, 76, 95, 48, 57, 65, 90, 
	97, 122, 46, 76, 95, 48, 57, 65, 
	90, 97, 122, 46, 78, 95, 48, 57, 
	65, 90, 97, 122, 46, 82, 95, 48, 
	57, 65, 90, 97, 122, 46, 85, 95, 
	48, 57, 65, 90, 97, 122, 46, 69, 
	95, 48, 57, 65, 90, 97, 122, 91, 
	46, 95, 114, 48, 57, 65, 90, 97, 
	122, 46, 95, 101, 48, 57, 65, 90, 
	97, 122, 46, 95, 97, 48, 57, 65, 
	90, 98, 122, 46, 95, 107, 48, 57, 
	65, 90, 97, 122, 46, 95, 108, 48, 
	57, 65, 90, 97, 122, 46, 95, 115, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	101, 48, 57, 65, 90, 97, 122, 46, 
	95, 111, 117, 48, 57, 65, 90, 97, 
	122, 46, 95, 114, 48, 57, 65, 90, 
	97, 122, 46, 95, 110, 48, 57, 65, 
	90, 97, 122, 46, 95, 99, 48, 57, 
	65, 90, 97, 122, 46, 95, 116, 48, 
	57, 65, 90, 97, 122, 46, 95, 105, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	111, 48, 57, 65, 90, 97, 122, 46, 
	95, 110, 48, 57, 65, 90, 97, 122, 
	46, 95, 102, 110, 48, 57, 65, 90, 
	97, 122, 46, 95, 101, 48, 57, 65, 
	90, 97, 122, 46, 95, 120, 48, 57, 
	65, 90, 97, 122, 46, 95, 116, 48, 
	57, 65, 90, 97, 122, 46, 95, 101, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	112, 48, 57, 65, 90, 97, 122, 46, 
	95, 101, 48, 57, 65, 90, 97, 122, 
	46, 95, 97, 48, 57, 65, 90, 98, 
	122, 46, 95, 116, 48, 57, 65, 90, 
	97, 122, 46, 95, 104, 48, 57, 65, 
	90, 97, 122, 46, 95, 105, 48, 57, 
	65, 90, 97, 122, 46, 95, 108, 48, 
	57, 65, 90, 97, 122, 46, 95, 101, 
	48, 57, 65, 90, 97, 122, 124, 0
};

static const char _Scanner_single_lengths[] = {
	0, 2, 1, 3, 0, 2, 3, 0, 
	2, 0, 1, 3, 0, 44, 2, 2, 
	1, 1, 1, 1, 1, 2, 2, 2, 
	4, 2, 5, 1, 1, 3, 1, 1, 
	3, 3, 3, 3, 3, 3, 5, 2, 
	5, 3, 3, 3, 3, 3, 3, 3, 
	3, 2, 3, 3, 3, 3, 3, 3, 
	2, 3, 3, 3, 2, 3, 3, 3, 
	3, 3, 3, 1, 3, 3, 3, 3, 
	3, 3, 3, 4, 3, 3, 3, 3, 
	3, 3, 3, 4, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	1
};

static const char _Scanner_range_lengths[] = {
	0, 1, 0, 0, 0, 0, 0, 0, 
	1, 1, 0, 0, 0, 2, 1, 1, 
	0, 0, 0, 0, 0, 3, 3, 3, 
	1, 1, 1, 0, 0, 0, 0, 0, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 0, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	0
};

static const short _Scanner_index_offsets[] = {
	0, 0, 4, 6, 10, 11, 14, 18, 
	19, 23, 25, 27, 31, 32, 79, 83, 
	87, 89, 91, 93, 95, 97, 103, 109, 
	115, 121, 125, 132, 134, 136, 140, 142, 
	144, 151, 158, 165, 172, 179, 186, 195, 
	201, 210, 217, 224, 231, 238, 245, 252, 
	259, 266, 272, 279, 286, 293, 300, 307, 
	314, 320, 327, 334, 341, 347, 354, 361, 
	368, 375, 382, 389, 391, 398, 405, 412, 
	419, 426, 433, 440, 448, 455, 462, 469, 
	476, 483, 490, 497, 505, 512, 519, 526, 
	533, 540, 547, 554, 561, 568, 575, 582, 
	589
};

static const unsigned char _Scanner_indicies[] = {
	2, 3, 0, 1, 2, 3, 6, 7, 
	8, 5, 5, 6, 10, 9, 6, 12, 
	13, 11, 11, 15, 15, 16, 14, 16, 
	14, 18, 17, 6, 20, 21, 19, 19, 
	2, 23, 5, 3, 24, 9, 25, 11, 
	26, 27, 28, 29, 30, 31, 32, 33, 
	35, 36, 37, 38, 39, 40, 41, 43, 
	44, 45, 46, 47, 6, 48, 49, 6, 
	19, 50, 51, 52, 53, 54, 55, 56, 
	57, 58, 59, 60, 34, 42, 22, 2, 
	3, 61, 22, 2, 3, 62, 1, 64, 
	63, 66, 65, 68, 67, 70, 69, 72, 
	71, 74, 42, 75, 42, 42, 73, 42, 
	42, 76, 42, 42, 4, 42, 42, 42, 
	42, 42, 4, 78, 79, 78, 80, 75, 
	77, 79, 80, 16, 77, 75, 78, 79, 
	78, 80, 34, 77, 82, 81, 84, 83, 
	86, 87, 88, 85, 90, 89, 92, 91, 
	42, 93, 42, 42, 42, 42, 73, 42, 
	94, 42, 42, 42, 42, 73, 42, 95, 
	42, 42, 42, 42, 73, 42, 96, 42, 
	42, 42, 42, 73, 42, 42, 97, 42, 
	42, 42, 73, 42, 42, 98, 42, 42, 
	42, 73, 42, 99, 100, 42, 101, 42, 
	42, 42, 73, 42, 103, 42, 42, 42, 
	102, 42, 42, 104, 105, 106, 42, 42, 
	42, 73, 42, 42, 107, 42, 42, 42, 
	73, 42, 42, 108, 42, 42, 42, 73, 
	42, 42, 109, 42, 42, 42, 73, 42, 
	42, 110, 42, 42, 42, 73, 42, 42, 
	111, 42, 42, 42, 73, 42, 42, 112, 
	42, 42, 42, 73, 42, 42, 113, 42, 
	42, 42, 73, 42, 42, 114, 42, 42, 
	42, 73, 42, 115, 42, 42, 42, 73, 
	42, 42, 116, 42, 42, 42, 73, 42, 
	42, 117, 42, 42, 42, 73, 42, 42, 
	118, 42, 42, 42, 73, 42, 42, 119, 
	42, 42, 42, 73, 42, 42, 120, 42, 
	42, 42, 73, 42, 42, 121, 42, 42, 
	42, 73, 42, 122, 42, 42, 42, 73, 
	42, 42, 123, 42, 42, 42, 73, 42, 
	42, 124, 42, 42, 42, 73, 42, 42, 
	125, 42, 42, 42, 73, 42, 126, 42, 
	42, 42, 73, 42, 127, 42, 42, 42, 
	42, 73, 42, 128, 42, 42, 42, 42, 
	73, 42, 129, 42, 42, 42, 42, 73, 
	42, 130, 42, 42, 42, 42, 73, 42, 
	131, 42, 42, 42, 42, 73, 42, 132, 
	42, 42, 42, 42, 73, 134, 133, 42, 
	42, 135, 42, 42, 42, 73, 42, 42, 
	136, 42, 42, 42, 73, 42, 42, 137, 
	42, 42, 42, 73, 42, 42, 138, 42, 
	42, 42, 73, 42, 42, 139, 42, 42, 
	42, 73, 42, 42, 140, 42, 42, 42, 
	73, 42, 42, 141, 42, 42, 42, 73, 
	42, 42, 142, 143, 42, 42, 42, 73, 
	42, 42, 144, 42, 42, 42, 73, 42, 
	42, 145, 42, 42, 42, 73, 42, 42, 
	146, 42, 42, 42, 73, 42, 42, 147, 
	42, 42, 42, 73, 42, 42, 148, 42, 
	42, 42, 73, 42, 42, 149, 42, 42, 
	42, 73, 42, 42, 150, 42, 42, 42, 
	73, 42, 42, 151, 152, 42, 42, 42, 
	73, 42, 42, 153, 42, 42, 42, 73, 
	42, 42, 154, 42, 42, 42, 73, 42, 
	42, 155, 42, 42, 42, 73, 42, 42, 
	156, 42, 42, 42, 73, 42, 42, 157, 
	42, 42, 42, 73, 42, 42, 158, 42, 
	42, 42, 73, 42, 42, 159, 42, 42, 
	42, 73, 42, 42, 160, 42, 42, 42, 
	73, 42, 42, 161, 42, 42, 42, 73, 
	42, 42, 162, 42, 42, 42, 73, 42, 
	42, 163, 42, 42, 42, 73, 42, 42, 
	164, 42, 42, 42, 73, 166, 165, 0
};

static const char _Scanner_trans_targs[] = {
	13, 1, 15, 2, 13, 3, 0, 13, 
	4, 5, 13, 6, 13, 7, 13, 9, 
	25, 13, 13, 11, 12, 13, 14, 16, 
	13, 17, 13, 13, 18, 13, 13, 19, 
	21, 13, 26, 27, 13, 29, 30, 31, 
	13, 13, 23, 32, 36, 38, 64, 67, 
	13, 13, 68, 72, 75, 83, 84, 87, 
	92, 13, 96, 13, 13, 13, 13, 13, 
	13, 13, 13, 13, 13, 13, 20, 13, 
	13, 13, 22, 24, 22, 13, 8, 13, 
	13, 13, 28, 13, 13, 13, 13, 10, 
	13, 13, 13, 13, 13, 33, 34, 35, 
	23, 37, 23, 39, 61, 63, 13, 40, 
	41, 50, 57, 42, 43, 44, 45, 46, 
	47, 48, 49, 23, 51, 52, 53, 54, 
	55, 56, 23, 58, 59, 60, 23, 62, 
	23, 23, 65, 66, 23, 13, 13, 69, 
	70, 71, 23, 73, 74, 23, 76, 77, 
	23, 78, 79, 80, 81, 82, 23, 23, 
	23, 85, 86, 23, 88, 89, 90, 91, 
	23, 93, 94, 95, 23, 13, 13
};

static const unsigned char _Scanner_trans_actions[] = {
	107, 0, 171, 0, 109, 0, 0, 7, 
	0, 0, 63, 0, 5, 0, 103, 0, 
	0, 105, 59, 0, 0, 9, 174, 0, 
	25, 0, 35, 37, 0, 15, 65, 0, 
	0, 19, 3, 0, 67, 3, 0, 0, 
	61, 27, 168, 0, 0, 0, 0, 0, 
	41, 17, 0, 0, 0, 0, 0, 0, 
	0, 31, 0, 33, 23, 101, 99, 81, 
	49, 87, 51, 79, 21, 77, 0, 97, 
	57, 71, 168, 3, 165, 73, 0, 11, 
	13, 83, 0, 85, 29, 93, 55, 0, 
	43, 75, 47, 95, 45, 0, 0, 0, 
	120, 0, 123, 0, 0, 0, 69, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 135, 0, 0, 0, 0, 
	0, 0, 129, 0, 0, 0, 132, 0, 
	114, 126, 0, 0, 117, 91, 39, 0, 
	0, 0, 162, 0, 0, 156, 0, 0, 
	147, 0, 0, 0, 0, 0, 138, 150, 
	153, 0, 0, 159, 0, 0, 0, 0, 
	144, 0, 0, 0, 141, 89, 53
};

static const unsigned char _Scanner_to_state_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 111, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0
};

static const unsigned char _Scanner_from_state_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 1, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0
};

static const short _Scanner_eof_trans[] = {
	0, 1, 5, 0, 0, 0, 0, 0, 
	15, 15, 18, 0, 0, 0, 62, 63, 
	64, 66, 68, 70, 72, 74, 5, 5, 
	78, 78, 78, 82, 84, 86, 90, 92, 
	74, 74, 74, 74, 74, 74, 74, 103, 
	74, 74, 74, 74, 74, 74, 74, 74, 
	74, 74, 74, 74, 74, 74, 74, 74, 
	74, 74, 74, 74, 74, 74, 74, 74, 
	74, 74, 74, 134, 74, 74, 74, 74, 
	74, 74, 74, 74, 74, 74, 74, 74, 
	74, 74, 74, 74, 74, 74, 74, 74, 
	74, 74, 74, 74, 74, 74, 74, 74, 
	166
};

static const int Scanner_start = 13;
static const int Scanner_first_final = 13;
static const int Scanner_error = 0;

static const int Scanner_en_main = 13;


#line 132 "r.rl"


void Parser::token(int tok, Value v)
{
	const char *data = ts;
	int len = te - ts;

	/*std::cout << '<' << tok << "> ";
	std::cout.write( data, len );
	std::cout << '\n';*/

	int initialErrors = errors;

	// Do the lookahead to resolve the dangling else conflict
	if(lastTokenWasNL) {
		if(tok != TOKEN_ELSE && (nesting.size()==0 || nesting.top()!=TOKEN_LPAREN))
			Parse(pParser, TOKEN_NEWLINE, Value::Nil(), this);
		Parse(pParser, tok, v, this);
		lastTokenWasNL = false;
	}
	else {
		if(tok == TOKEN_NEWLINE)
			lastTokenWasNL = true;
		else
			Parse(pParser, tok, v, this);
	}

	le = te;

	if(tok == TOKEN_LPAREN) nesting.push(tok);
	else if(tok == TOKEN_LBRACE) nesting.push(tok);
	else if(tok == TOKEN_RPAREN || tok == TOKEN_RBRACE) nesting.pop();
	else if(tok == TOKEN_FUNCTION) source.push(ts);

	/* Count newlines and columns. Use for error reporting? */ 
	for ( int i = 0; i < len; i ++ ) {
		if ( data[i] == '\n' ) {
			line += 1;
			col = 1;
		}
		else {
			col += 1;
		}
	}

	if(errors > initialErrors) {
		std::cout << "Error (" << intToStr(line+1) << "," << intToStr(col+1) << ") : unexpected '" << std::string(data, len) + "'" << std::endl; 
	}
}

Parser::Parser(State& state) : line(0), col(0), state(state), errors(0), complete(false), lastTokenWasNL(false) 
{}

int Parser::execute( const char* data, int len, bool isEof, Value& out, FILE* trace )
{
	out = Value::Nil();
	errors = 0;
	lastTokenWasNL = false;
	complete = false;

	pParser = ParseAlloc(malloc);

	/*ParseTrace(trace, 0);*/

	const char *p = data;
	const char *pe = data+len;
	const char* eof = isEof ? pe : 0;
	int cs, act;
	
#line 453 "parser.cpp"
	{
	cs = Scanner_start;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 201 "r.rl"
	
#line 463 "parser.cpp"
	{
	int _klen;
	unsigned int _trans;
	const char *_acts;
	unsigned int _nacts;
	const char *_keys;

	if ( p == pe )
		goto _test_eof;
	if ( cs == 0 )
		goto _out;
_resume:
	_acts = _Scanner_actions + _Scanner_from_state_actions[cs];
	_nacts = (unsigned int) *_acts++;
	while ( _nacts-- > 0 ) {
		switch ( *_acts++ ) {
	case 2:
#line 1 "NONE"
	{ts = p;}
	break;
#line 484 "parser.cpp"
		}
	}

	_keys = _Scanner_trans_keys + _Scanner_key_offsets[cs];
	_trans = _Scanner_index_offsets[cs];

	_klen = _Scanner_single_lengths[cs];
	if ( _klen > 0 ) {
		const char *_lower = _keys;
		const char *_mid;
		const char *_upper = _keys + _klen - 1;
		while (1) {
			if ( _upper < _lower )
				break;

			_mid = _lower + ((_upper-_lower) >> 1);
			if ( (*p) < *_mid )
				_upper = _mid - 1;
			else if ( (*p) > *_mid )
				_lower = _mid + 1;
			else {
				_trans += (_mid - _keys);
				goto _match;
			}
		}
		_keys += _klen;
		_trans += _klen;
	}

	_klen = _Scanner_range_lengths[cs];
	if ( _klen > 0 ) {
		const char *_lower = _keys;
		const char *_mid;
		const char *_upper = _keys + (_klen<<1) - 2;
		while (1) {
			if ( _upper < _lower )
				break;

			_mid = _lower + (((_upper-_lower) >> 1) & ~1);
			if ( (*p) < _mid[0] )
				_upper = _mid - 2;
			else if ( (*p) > _mid[1] )
				_lower = _mid + 2;
			else {
				_trans += ((_mid - _keys)>>1);
				goto _match;
			}
		}
		_trans += _klen;
	}

_match:
	_trans = _Scanner_indicies[_trans];
_eof_trans:
	cs = _Scanner_trans_targs[_trans];

	if ( _Scanner_trans_actions[_trans] == 0 )
		goto _again;

	_acts = _Scanner_actions + _Scanner_trans_actions[_trans];
	_nacts = (unsigned int) *_acts++;
	while ( _nacts-- > 0 )
	{
		switch ( *_acts++ )
		{
	case 3:
#line 1 "NONE"
	{te = p+1;}
	break;
	case 4:
#line 29 "r.rl"
	{act = 1;}
	break;
	case 5:
#line 31 "r.rl"
	{act = 3;}
	break;
	case 6:
#line 32 "r.rl"
	{act = 4;}
	break;
	case 7:
#line 33 "r.rl"
	{act = 5;}
	break;
	case 8:
#line 34 "r.rl"
	{act = 6;}
	break;
	case 9:
#line 35 "r.rl"
	{act = 7;}
	break;
	case 10:
#line 36 "r.rl"
	{act = 8;}
	break;
	case 11:
#line 37 "r.rl"
	{act = 9;}
	break;
	case 12:
#line 39 "r.rl"
	{act = 10;}
	break;
	case 13:
#line 40 "r.rl"
	{act = 11;}
	break;
	case 14:
#line 41 "r.rl"
	{act = 12;}
	break;
	case 15:
#line 42 "r.rl"
	{act = 13;}
	break;
	case 16:
#line 43 "r.rl"
	{act = 14;}
	break;
	case 17:
#line 44 "r.rl"
	{act = 15;}
	break;
	case 18:
#line 45 "r.rl"
	{act = 16;}
	break;
	case 19:
#line 46 "r.rl"
	{act = 17;}
	break;
	case 20:
#line 47 "r.rl"
	{act = 18;}
	break;
	case 21:
#line 57 "r.rl"
	{act = 21;}
	break;
	case 22:
#line 60 "r.rl"
	{act = 22;}
	break;
	case 23:
#line 126 "r.rl"
	{act = 66;}
	break;
	case 24:
#line 129 "r.rl"
	{act = 67;}
	break;
	case 25:
#line 51 "r.rl"
	{te = p+1;{std::string s(ts+1, te-ts-2); token( TOKEN_STR_CONST, Character::c(state.internStr(unescape(s))) );}}
	break;
	case 26:
#line 53 "r.rl"
	{te = p+1;{std::string s(ts+1, te-ts-2); token( TOKEN_STR_CONST, Character::c(state.internStr(unescape(s))) );}}
	break;
	case 27:
#line 62 "r.rl"
	{te = p+1;{std::string s(ts+1, te-ts-2); token( TOKEN_SYMBOL, CreateSymbol(state.internStr(unescape(s))) );}}
	break;
	case 28:
#line 68 "r.rl"
	{te = p+1;{token( TOKEN_NUM_CONST, Integer::c(strToDouble(std::string(ts, te-ts-1).c_str())) );}}
	break;
	case 29:
#line 71 "r.rl"
	{te = p+1;{token( TOKEN_NUM_CONST, CreateComplex(strToDouble(std::string(ts, te-ts-1))) );}}
	break;
	case 30:
#line 83 "r.rl"
	{te = p+1;{token( TOKEN_PLUS, CreateSymbol(Strings::add) );}}
	break;
	case 31:
#line 85 "r.rl"
	{te = p+1;{token( TOKEN_POW, CreateSymbol(Strings::pow) );}}
	break;
	case 32:
#line 86 "r.rl"
	{te = p+1;{token( TOKEN_DIVIDE, CreateSymbol(Strings::div) );}}
	break;
	case 33:
#line 88 "r.rl"
	{te = p+1;{token( TOKEN_POW, CreateSymbol(Strings::pow) );}}
	break;
	case 34:
#line 89 "r.rl"
	{te = p+1;{token( TOKEN_TILDE, CreateSymbol(Strings::tilde) );}}
	break;
	case 35:
#line 90 "r.rl"
	{te = p+1;{token( TOKEN_DOLLAR, CreateSymbol(Strings::dollar) );}}
	break;
	case 36:
#line 91 "r.rl"
	{te = p+1;{token( TOKEN_AT, CreateSymbol(Strings::at) );}}
	break;
	case 37:
#line 95 "r.rl"
	{te = p+1;{token( TOKEN_NS_GET_INT, CreateSymbol(Strings::nsgetint) );}}
	break;
	case 38:
#line 98 "r.rl"
	{te = p+1;{token( TOKEN_LBRACE, CreateSymbol(Strings::brace) );}}
	break;
	case 39:
#line 99 "r.rl"
	{te = p+1;{token( TOKEN_RBRACE );}}
	break;
	case 40:
#line 100 "r.rl"
	{te = p+1;{token( TOKEN_LPAREN, CreateSymbol(Strings::paren) );}}
	break;
	case 41:
#line 101 "r.rl"
	{te = p+1;{token( TOKEN_RPAREN );}}
	break;
	case 42:
#line 103 "r.rl"
	{te = p+1;{token( TOKEN_LBB, CreateSymbol(Strings::bb) );}}
	break;
	case 43:
#line 104 "r.rl"
	{te = p+1;{token( TOKEN_RBRACKET );}}
	break;
	case 44:
#line 107 "r.rl"
	{te = p+1;{token( TOKEN_LE, CreateSymbol(Strings::le) );}}
	break;
	case 45:
#line 108 "r.rl"
	{te = p+1;{token( TOKEN_GE, CreateSymbol(Strings::ge) );}}
	break;
	case 46:
#line 109 "r.rl"
	{te = p+1;{token( TOKEN_EQ, CreateSymbol(Strings::eq) );}}
	break;
	case 47:
#line 110 "r.rl"
	{te = p+1;{token( TOKEN_NE, CreateSymbol(Strings::neq) );}}
	break;
	case 48:
#line 111 "r.rl"
	{te = p+1;{token( TOKEN_AND2, CreateSymbol(Strings::land2) );}}
	break;
	case 49:
#line 112 "r.rl"
	{te = p+1;{token( TOKEN_OR2, CreateSymbol(Strings::lor2) );}}
	break;
	case 50:
#line 113 "r.rl"
	{te = p+1;{token( TOKEN_LEFT_ASSIGN, CreateSymbol(Strings::assign) );}}
	break;
	case 51:
#line 115 "r.rl"
	{te = p+1;{token( TOKEN_RIGHT_ASSIGN, CreateSymbol(Strings::assign2) );}}
	break;
	case 52:
#line 116 "r.rl"
	{te = p+1;{token( TOKEN_LEFT_ASSIGN, CreateSymbol(Strings::assign2) );}}
	break;
	case 53:
#line 117 "r.rl"
	{te = p+1;{token( TOKEN_QUESTION, CreateSymbol(Strings::question) );}}
	break;
	case 54:
#line 120 "r.rl"
	{te = p+1;{token(TOKEN_SPECIALOP, CreateSymbol(state.internStr(std::string(ts, te-ts))) ); }}
	break;
	case 55:
#line 123 "r.rl"
	{te = p+1;{token( TOKEN_COMMA );}}
	break;
	case 56:
#line 124 "r.rl"
	{te = p+1;{token( TOKEN_SEMICOLON );}}
	break;
	case 57:
#line 30 "r.rl"
	{te = p;p--;{token( TOKEN_NUM_CONST, Logical::NA() );}}
	break;
	case 58:
#line 60 "r.rl"
	{te = p;p--;{token( TOKEN_SYMBOL, CreateSymbol(state.internStr(std::string(ts, te-ts))) );}}
	break;
	case 59:
#line 65 "r.rl"
	{te = p;p--;{token( TOKEN_NUM_CONST, Double::c(strToDouble(std::string(ts, te-ts).c_str())) );}}
	break;
	case 60:
#line 82 "r.rl"
	{te = p;p--;{token( TOKEN_EQ_ASSIGN, CreateSymbol(Strings::eqassign) );}}
	break;
	case 61:
#line 84 "r.rl"
	{te = p;p--;{token( TOKEN_MINUS, CreateSymbol(Strings::sub) );}}
	break;
	case 62:
#line 87 "r.rl"
	{te = p;p--;{token( TOKEN_TIMES, CreateSymbol(Strings::mul) );}}
	break;
	case 63:
#line 92 "r.rl"
	{te = p;p--;{token( TOKEN_NOT, CreateSymbol(Strings::lnot) );}}
	break;
	case 64:
#line 93 "r.rl"
	{te = p;p--;{token( TOKEN_COLON, CreateSymbol(Strings::colon) );}}
	break;
	case 65:
#line 94 "r.rl"
	{te = p;p--;{token( TOKEN_NS_GET, CreateSymbol(Strings::nsget) );}}
	break;
	case 66:
#line 96 "r.rl"
	{te = p;p--;{token( TOKEN_AND, CreateSymbol(Strings::land) );}}
	break;
	case 67:
#line 97 "r.rl"
	{te = p;p--;{token( TOKEN_OR, CreateSymbol(Strings::lor) );}}
	break;
	case 68:
#line 102 "r.rl"
	{te = p;p--;{token( TOKEN_LBRACKET, CreateSymbol(Strings::bracket) );}}
	break;
	case 69:
#line 105 "r.rl"
	{te = p;p--;{token( TOKEN_LT, CreateSymbol(Strings::lt) );}}
	break;
	case 70:
#line 106 "r.rl"
	{te = p;p--;{token( TOKEN_GT, CreateSymbol(Strings::gt) );}}
	break;
	case 71:
#line 114 "r.rl"
	{te = p;p--;{token( TOKEN_RIGHT_ASSIGN, CreateSymbol(Strings::assign) );}}
	break;
	case 72:
#line 126 "r.rl"
	{te = p;p--;{token( TOKEN_NEWLINE );}}
	break;
	case 73:
#line 129 "r.rl"
	{te = p;p--;}
	break;
	case 74:
#line 65 "r.rl"
	{{p = ((te))-1;}{token( TOKEN_NUM_CONST, Double::c(strToDouble(std::string(ts, te-ts).c_str())) );}}
	break;
	case 75:
#line 105 "r.rl"
	{{p = ((te))-1;}{token( TOKEN_LT, CreateSymbol(Strings::lt) );}}
	break;
	case 76:
#line 126 "r.rl"
	{{p = ((te))-1;}{token( TOKEN_NEWLINE );}}
	break;
	case 77:
#line 1 "NONE"
	{	switch( act ) {
	case 0:
	{{cs = 0; goto _again;}}
	break;
	case 1:
	{{p = ((te))-1;}token( TOKEN_NULL_CONST, Null::Singleton() );}
	break;
	case 3:
	{{p = ((te))-1;}token( TOKEN_NUM_CONST, Logical::True() );}
	break;
	case 4:
	{{p = ((te))-1;}token( TOKEN_NUM_CONST, Logical::False() );}
	break;
	case 5:
	{{p = ((te))-1;}token( TOKEN_NUM_CONST, Double::Inf() );}
	break;
	case 6:
	{{p = ((te))-1;}token( TOKEN_NUM_CONST, Double::NaN() );}
	break;
	case 7:
	{{p = ((te))-1;}token( TOKEN_NUM_CONST, Integer::NA() );}
	break;
	case 8:
	{{p = ((te))-1;}token( TOKEN_NUM_CONST, Double::NA() );}
	break;
	case 9:
	{{p = ((te))-1;}token( TOKEN_STR_CONST, Character::NA() );}
	break;
	case 10:
	{{p = ((te))-1;}token( TOKEN_FUNCTION, CreateSymbol(Strings::function) );}
	break;
	case 11:
	{{p = ((te))-1;}token( TOKEN_WHILE, CreateSymbol(Strings::whileSym) );}
	break;
	case 12:
	{{p = ((te))-1;}token( TOKEN_REPEAT, CreateSymbol(Strings::repeatSym) );}
	break;
	case 13:
	{{p = ((te))-1;}token( TOKEN_FOR, CreateSymbol(Strings::forSym) );}
	break;
	case 14:
	{{p = ((te))-1;}token( TOKEN_IF, CreateSymbol(Strings::ifSym) );}
	break;
	case 15:
	{{p = ((te))-1;}token( TOKEN_IN );}
	break;
	case 16:
	{{p = ((te))-1;}token( TOKEN_ELSE );}
	break;
	case 17:
	{{p = ((te))-1;}token( TOKEN_NEXT, CreateSymbol(Strings::nextSym) );}
	break;
	case 18:
	{{p = ((te))-1;}token( TOKEN_BREAK, CreateSymbol(Strings::breakSym) );}
	break;
	case 21:
	{{p = ((te))-1;}token( TOKEN_SYMBOL, CreateSymbol(state.internStr(std::string(ts, te-ts))));}
	break;
	case 22:
	{{p = ((te))-1;}token( TOKEN_SYMBOL, CreateSymbol(state.internStr(std::string(ts, te-ts))) );}
	break;
	case 66:
	{{p = ((te))-1;}token( TOKEN_NEWLINE );}
	break;
	default:
	{{p = ((te))-1;}}
	break;
	}
	}
	break;
#line 918 "parser.cpp"
		}
	}

_again:
	_acts = _Scanner_actions + _Scanner_to_state_actions[cs];
	_nacts = (unsigned int) *_acts++;
	while ( _nacts-- > 0 ) {
		switch ( *_acts++ ) {
	case 0:
#line 1 "NONE"
	{ts = 0;}
	break;
	case 1:
#line 1 "NONE"
	{act = 0;}
	break;
#line 935 "parser.cpp"
		}
	}

	if ( cs == 0 )
		goto _out;
	if ( ++p != pe )
		goto _resume;
	_test_eof: {}
	if ( p == eof )
	{
	if ( _Scanner_eof_trans[cs] > 0 ) {
		_trans = _Scanner_eof_trans[cs] - 1;
		goto _eof_trans;
	}
	}

	_out: {}
	}

#line 202 "r.rl"
	int syntaxErrors = errors;
	Parse(pParser, 0, Value::Nil(), this);
	ParseFree(pParser, free);
	errors = syntaxErrors;

	if( cs == Scanner_error && syntaxErrors == 0) {
		syntaxErrors++;
		std::cout << "Lexing error (" << intToStr(line+1) << "," << intToStr(col+1) << ") : unexpected '" << std::string(ts, te-ts) + "'" << std::endl; 
	}
	
	if( syntaxErrors > 0 )
		return -1;
	else if( cs >= Scanner_first_final && complete) {
		out = result;
		return 1;
	} 
	else
		return 0;
}
/*
int Parser::buffer_execute( )
{
	static char buf[16384];

	std::ios::sync_with_stdio(false);

	bool done = false;
	while ( !done ) {
		char* b = buf + have;
		const char *p = b;
		int space = 16384 - have;

		if ( space == 0 ) {
			std::cerr << "OUT OF BUFFER SPACE" << std::endl;
			return -1;
		}

		std::cin.read( b, space );
		int len = std::cin.gcount();
		const char *pe = p + len;
		const char *eof = 0;

	 	if ( std::cin.eof() ) {
			eof = pe;
			done = true;
		}

		%% write exec;

		if ( cs == Scanner_error ) {
			std::cerr << "PARSE ERROR" << std::endl;
			return -1;
		}

		if ( ts == 0 )
			have = 0;
		else {
			have = pe - ts;
			memmove( buf, ts, have );
			te -= (ts-buf);
			ts = buf;
		}
	}
	return 0;
}
*/


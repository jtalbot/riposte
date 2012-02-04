
#line 1 "lexer.rl"
/*
 *	A ragel lexer for R.
 *	In ragel, the lexer drives the parsing process, so this also has the basic parsing functions.
 *	Use this to generate parser.cpp
 *      TODO: 
 *            Parse escape sequences embedded in strings.
 *            Include the double-to-int warnings, e.g. on 1.0L  and 1.5L
 *            Generate hex numbers
 *            Do we really want to allow '.' inside hex numbers? R allows them, but ignores them when parsing to a number.
 *            R parser has a rule for OP % OP. Is this ever used?
 */

#include "parser.h"
#include "tokens.h"
#include "ast.cpp"


#line 21 "../parser.cpp"
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
	74, 1, 75, 1, 76, 1, 77, 1, 
	78, 2, 0, 1, 2, 3, 4, 2, 
	3, 5, 2, 3, 6, 2, 3, 7, 
	2, 3, 8, 2, 3, 9, 2, 3, 
	10, 2, 3, 11, 2, 3, 12, 2, 
	3, 13, 2, 3, 14, 2, 3, 15, 
	2, 3, 16, 2, 3, 17, 2, 3, 
	18, 2, 3, 19, 2, 3, 20, 2, 
	3, 21, 2, 3, 22, 2, 3, 23, 
	2, 3, 24
};

static const short _Scanner_key_offsets[] = {
	0, 0, 4, 5, 8, 8, 10, 13, 
	13, 17, 19, 25, 29, 31, 32, 35, 
	35, 84, 88, 92, 93, 94, 95, 96, 
	97, 105, 113, 121, 126, 129, 136, 142, 
	150, 152, 153, 154, 157, 158, 159, 168, 
	177, 186, 195, 204, 213, 224, 232, 243, 
	252, 261, 270, 279, 288, 297, 306, 315, 
	323, 332, 341, 350, 359, 368, 377, 385, 
	394, 403, 412, 420, 429, 438, 447, 456, 
	465, 474, 475, 484, 493, 502, 511, 520, 
	529, 538, 548, 557, 566, 575, 584, 593, 
	602, 611, 621, 630, 639, 648, 657, 666, 
	675, 684, 693, 702, 711, 720, 729
};

static const char _Scanner_trans_keys[] = {
	10, 35, 33, 126, 10, 10, 34, 92, 
	10, 37, 10, 39, 92, 43, 45, 48, 
	57, 48, 57, 48, 57, 65, 70, 97, 
	102, 43, 45, 48, 57, 48, 57, 45, 
	10, 92, 96, 10, 33, 34, 35, 36, 
	37, 38, 39, 40, 41, 42, 43, 44, 
	45, 46, 47, 48, 58, 59, 60, 61, 
	62, 63, 64, 70, 73, 78, 84, 91, 
	92, 93, 94, 95, 96, 98, 101, 102, 
	105, 110, 114, 119, 123, 124, 125, 126, 
	49, 57, 65, 122, 10, 35, 33, 126, 
	10, 35, 33, 126, 61, 38, 42, 62, 
	62, 46, 95, 48, 57, 65, 90, 97, 
	122, 46, 95, 48, 57, 65, 90, 97, 
	122, 46, 95, 48, 57, 65, 90, 97, 
	122, 69, 76, 101, 48, 57, 76, 48, 
	57, 46, 69, 76, 101, 120, 48, 57, 
	46, 69, 76, 101, 48, 57, 80, 112, 
	48, 57, 65, 70, 97, 102, 48, 57, 
	58, 58, 45, 60, 61, 61, 61, 46, 
	65, 95, 48, 57, 66, 90, 97, 122, 
	46, 76, 95, 48, 57, 65, 90, 97, 
	122, 46, 83, 95, 48, 57, 65, 90, 
	97, 122, 46, 69, 95, 48, 57, 65, 
	90, 97, 122, 46, 95, 110, 48, 57, 
	65, 90, 97, 122, 46, 95, 102, 48, 
	57, 65, 90, 97, 122, 46, 65, 85, 
	95, 97, 48, 57, 66, 90, 98, 122, 
	46, 95, 48, 57, 65, 90, 97, 122, 
	46, 95, 99, 105, 114, 48, 57, 65, 
	90, 97, 122, 46, 95, 104, 48, 57, 
	65, 90, 97, 122, 46, 95, 97, 48, 
	57, 65, 90, 98, 122, 46, 95, 114, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	97, 48, 57, 65, 90, 98, 122, 46, 
	95, 99, 48, 57, 65, 90, 97, 122, 
	46, 95, 116, 48, 57, 65, 90, 97, 
	122, 46, 95, 101, 48, 57, 65, 90, 
	97, 122, 46, 95, 114, 48, 57, 65, 
	90, 97, 122, 46, 95, 48, 57, 65, 
	90, 97, 122, 46, 95, 110, 48, 57, 
	65, 90, 97, 122, 46, 95, 116, 48, 
	57, 65, 90, 97, 122, 46, 95, 101, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	103, 48, 57, 65, 90, 97, 122, 46, 
	95, 101, 48, 57, 65, 90, 97, 122, 
	46, 95, 114, 48, 57, 65, 90, 97, 
	122, 46, 95, 48, 57, 65, 90, 97, 
	122, 46, 95, 101, 48, 57, 65, 90, 
	97, 122, 46, 95, 97, 48, 57, 65, 
	90, 98, 122, 46, 95, 108, 48, 57, 
	65, 90, 97, 122, 46, 95, 48, 57, 
	65, 90, 97, 122, 46, 76, 95, 48, 
	57, 65, 90, 97, 122, 46, 76, 95, 
	48, 57, 65, 90, 97, 122, 46, 78, 
	95, 48, 57, 65, 90, 97, 122, 46, 
	82, 95, 48, 57, 65, 90, 97, 122, 
	46, 85, 95, 48, 57, 65, 90, 97, 
	122, 46, 69, 95, 48, 57, 65, 90, 
	97, 122, 91, 46, 95, 114, 48, 57, 
	65, 90, 97, 122, 46, 95, 101, 48, 
	57, 65, 90, 97, 122, 46, 95, 97, 
	48, 57, 65, 90, 98, 122, 46, 95, 
	107, 48, 57, 65, 90, 97, 122, 46, 
	95, 108, 48, 57, 65, 90, 97, 122, 
	46, 95, 115, 48, 57, 65, 90, 97, 
	122, 46, 95, 101, 48, 57, 65, 90, 
	97, 122, 46, 95, 111, 117, 48, 57, 
	65, 90, 97, 122, 46, 95, 114, 48, 
	57, 65, 90, 97, 122, 46, 95, 110, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	99, 48, 57, 65, 90, 97, 122, 46, 
	95, 116, 48, 57, 65, 90, 97, 122, 
	46, 95, 105, 48, 57, 65, 90, 97, 
	122, 46, 95, 111, 48, 57, 65, 90, 
	97, 122, 46, 95, 110, 48, 57, 65, 
	90, 97, 122, 46, 95, 102, 110, 48, 
	57, 65, 90, 97, 122, 46, 95, 101, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	120, 48, 57, 65, 90, 97, 122, 46, 
	95, 116, 48, 57, 65, 90, 97, 122, 
	46, 95, 101, 48, 57, 65, 90, 97, 
	122, 46, 95, 112, 48, 57, 65, 90, 
	97, 122, 46, 95, 101, 48, 57, 65, 
	90, 97, 122, 46, 95, 97, 48, 57, 
	65, 90, 98, 122, 46, 95, 116, 48, 
	57, 65, 90, 97, 122, 46, 95, 104, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	105, 48, 57, 65, 90, 97, 122, 46, 
	95, 108, 48, 57, 65, 90, 97, 122, 
	46, 95, 101, 48, 57, 65, 90, 97, 
	122, 124, 0
};

static const char _Scanner_single_lengths[] = {
	0, 2, 1, 3, 0, 2, 3, 0, 
	2, 0, 0, 2, 0, 1, 3, 0, 
	45, 2, 2, 1, 1, 1, 1, 1, 
	2, 2, 2, 3, 1, 5, 4, 2, 
	0, 1, 1, 3, 1, 1, 3, 3, 
	3, 3, 3, 3, 5, 2, 5, 3, 
	3, 3, 3, 3, 3, 3, 3, 2, 
	3, 3, 3, 3, 3, 3, 2, 3, 
	3, 3, 2, 3, 3, 3, 3, 3, 
	3, 1, 3, 3, 3, 3, 3, 3, 
	3, 4, 3, 3, 3, 3, 3, 3, 
	3, 4, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 1
};

static const char _Scanner_range_lengths[] = {
	0, 1, 0, 0, 0, 0, 0, 0, 
	1, 1, 3, 1, 1, 0, 0, 0, 
	2, 1, 1, 0, 0, 0, 0, 0, 
	3, 3, 3, 1, 1, 1, 1, 3, 
	1, 0, 0, 0, 0, 0, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 0, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 0
};

static const short _Scanner_index_offsets[] = {
	0, 0, 4, 6, 10, 11, 14, 18, 
	19, 23, 25, 29, 33, 35, 37, 41, 
	42, 90, 94, 98, 100, 102, 104, 106, 
	108, 114, 120, 126, 131, 134, 141, 147, 
	153, 155, 157, 159, 163, 165, 167, 174, 
	181, 188, 195, 202, 209, 218, 224, 233, 
	240, 247, 254, 261, 268, 275, 282, 289, 
	295, 302, 309, 316, 323, 330, 337, 343, 
	350, 357, 364, 370, 377, 384, 391, 398, 
	405, 412, 414, 421, 428, 435, 442, 449, 
	456, 463, 471, 478, 485, 492, 499, 506, 
	513, 520, 528, 535, 542, 549, 556, 563, 
	570, 577, 584, 591, 598, 605, 612
};

static const unsigned char _Scanner_indicies[] = {
	2, 3, 0, 1, 2, 3, 6, 7, 
	8, 5, 5, 6, 10, 9, 6, 12, 
	13, 11, 11, 15, 15, 16, 14, 16, 
	14, 17, 17, 17, 14, 19, 19, 20, 
	18, 20, 18, 22, 21, 6, 24, 25, 
	23, 23, 2, 27, 5, 3, 28, 9, 
	29, 11, 30, 31, 32, 33, 34, 35, 
	36, 37, 38, 40, 41, 42, 43, 44, 
	45, 46, 48, 49, 50, 51, 52, 6, 
	53, 54, 6, 23, 55, 56, 57, 58, 
	59, 60, 61, 62, 63, 64, 65, 39, 
	47, 26, 2, 3, 66, 26, 2, 3, 
	67, 1, 69, 68, 71, 70, 73, 72, 
	75, 74, 77, 76, 79, 47, 80, 47, 
	47, 78, 47, 47, 81, 47, 47, 4, 
	47, 47, 47, 47, 47, 4, 83, 84, 
	83, 80, 82, 84, 16, 82, 80, 83, 
	84, 83, 85, 39, 82, 80, 83, 84, 
	83, 39, 82, 87, 87, 17, 17, 17, 
	86, 20, 86, 89, 88, 91, 90, 93, 
	94, 95, 92, 97, 96, 99, 98, 47, 
	100, 47, 47, 47, 47, 78, 47, 101, 
	47, 47, 47, 47, 78, 47, 102, 47, 
	47, 47, 47, 78, 47, 103, 47, 47, 
	47, 47, 78, 47, 47, 104, 47, 47, 
	47, 78, 47, 47, 105, 47, 47, 47, 
	78, 47, 106, 107, 47, 108, 47, 47, 
	47, 78, 47, 110, 47, 47, 47, 109, 
	47, 47, 111, 112, 113, 47, 47, 47, 
	78, 47, 47, 114, 47, 47, 47, 78, 
	47, 47, 115, 47, 47, 47, 78, 47, 
	47, 116, 47, 47, 47, 78, 47, 47, 
	117, 47, 47, 47, 78, 47, 47, 118, 
	47, 47, 47, 78, 47, 47, 119, 47, 
	47, 47, 78, 47, 47, 120, 47, 47, 
	47, 78, 47, 47, 121, 47, 47, 47, 
	78, 47, 122, 47, 47, 47, 78, 47, 
	47, 123, 47, 47, 47, 78, 47, 47, 
	124, 47, 47, 47, 78, 47, 47, 125, 
	47, 47, 47, 78, 47, 47, 126, 47, 
	47, 47, 78, 47, 47, 127, 47, 47, 
	47, 78, 47, 47, 128, 47, 47, 47, 
	78, 47, 129, 47, 47, 47, 78, 47, 
	47, 130, 47, 47, 47, 78, 47, 47, 
	131, 47, 47, 47, 78, 47, 47, 132, 
	47, 47, 47, 78, 47, 133, 47, 47, 
	47, 78, 47, 134, 47, 47, 47, 47, 
	78, 47, 135, 47, 47, 47, 47, 78, 
	47, 136, 47, 47, 47, 47, 78, 47, 
	137, 47, 47, 47, 47, 78, 47, 138, 
	47, 47, 47, 47, 78, 47, 139, 47, 
	47, 47, 47, 78, 141, 140, 47, 47, 
	142, 47, 47, 47, 78, 47, 47, 143, 
	47, 47, 47, 78, 47, 47, 144, 47, 
	47, 47, 78, 47, 47, 145, 47, 47, 
	47, 78, 47, 47, 146, 47, 47, 47, 
	78, 47, 47, 147, 47, 47, 47, 78, 
	47, 47, 148, 47, 47, 47, 78, 47, 
	47, 149, 150, 47, 47, 47, 78, 47, 
	47, 151, 47, 47, 47, 78, 47, 47, 
	152, 47, 47, 47, 78, 47, 47, 153, 
	47, 47, 47, 78, 47, 47, 154, 47, 
	47, 47, 78, 47, 47, 155, 47, 47, 
	47, 78, 47, 47, 156, 47, 47, 47, 
	78, 47, 47, 157, 47, 47, 47, 78, 
	47, 47, 158, 159, 47, 47, 47, 78, 
	47, 47, 160, 47, 47, 47, 78, 47, 
	47, 161, 47, 47, 47, 78, 47, 47, 
	162, 47, 47, 47, 78, 47, 47, 163, 
	47, 47, 47, 78, 47, 47, 164, 47, 
	47, 47, 78, 47, 47, 165, 47, 47, 
	47, 78, 47, 47, 166, 47, 47, 47, 
	78, 47, 47, 167, 47, 47, 47, 78, 
	47, 47, 168, 47, 47, 47, 78, 47, 
	47, 169, 47, 47, 47, 78, 47, 47, 
	170, 47, 47, 47, 78, 47, 47, 171, 
	47, 47, 47, 78, 173, 172, 0
};

static const char _Scanner_trans_targs[] = {
	16, 1, 18, 2, 16, 3, 0, 16, 
	4, 5, 16, 6, 16, 7, 16, 9, 
	28, 31, 16, 12, 32, 16, 16, 14, 
	15, 16, 17, 19, 16, 20, 16, 16, 
	21, 16, 16, 22, 24, 16, 29, 30, 
	33, 16, 35, 36, 37, 16, 16, 26, 
	38, 42, 44, 70, 73, 16, 16, 74, 
	78, 81, 89, 90, 93, 98, 16, 102, 
	16, 16, 16, 16, 16, 16, 16, 16, 
	16, 16, 16, 23, 16, 16, 16, 25, 
	27, 25, 16, 8, 16, 10, 16, 11, 
	16, 34, 16, 16, 16, 16, 13, 16, 
	16, 16, 16, 16, 39, 40, 41, 26, 
	43, 26, 45, 67, 69, 16, 46, 47, 
	56, 63, 48, 49, 50, 51, 52, 53, 
	54, 55, 26, 57, 58, 59, 60, 61, 
	62, 26, 64, 65, 66, 26, 68, 26, 
	26, 71, 72, 26, 16, 16, 75, 76, 
	77, 26, 79, 80, 26, 82, 83, 26, 
	84, 85, 86, 87, 88, 26, 26, 26, 
	91, 92, 26, 94, 95, 96, 97, 26, 
	99, 100, 101, 26, 16, 16
};

static const unsigned char _Scanner_trans_actions[] = {
	109, 0, 173, 0, 111, 0, 0, 7, 
	0, 0, 61, 0, 5, 0, 103, 0, 
	0, 3, 105, 0, 0, 107, 57, 0, 
	0, 9, 176, 0, 23, 0, 33, 35, 
	0, 13, 63, 0, 0, 17, 3, 3, 
	0, 65, 3, 0, 0, 59, 25, 170, 
	0, 0, 0, 0, 0, 39, 15, 0, 
	0, 0, 0, 0, 0, 0, 29, 0, 
	31, 21, 101, 99, 81, 47, 87, 49, 
	79, 19, 77, 0, 97, 55, 69, 170, 
	3, 167, 71, 0, 11, 0, 73, 0, 
	83, 0, 85, 27, 93, 53, 0, 41, 
	75, 45, 95, 43, 0, 0, 0, 122, 
	0, 125, 0, 0, 0, 67, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 137, 0, 0, 0, 0, 0, 
	0, 131, 0, 0, 0, 134, 0, 116, 
	128, 0, 0, 119, 91, 37, 0, 0, 
	0, 164, 0, 0, 158, 0, 0, 149, 
	0, 0, 0, 0, 0, 140, 152, 155, 
	0, 0, 161, 0, 0, 0, 0, 146, 
	0, 0, 0, 143, 89, 51
};

static const unsigned char _Scanner_to_state_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	113, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0
};

static const unsigned char _Scanner_from_state_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0
};

static const short _Scanner_eof_trans[] = {
	0, 1, 5, 0, 0, 0, 0, 0, 
	15, 15, 15, 19, 19, 22, 0, 0, 
	0, 67, 68, 69, 71, 73, 75, 77, 
	79, 5, 5, 83, 83, 83, 83, 87, 
	87, 89, 91, 93, 97, 99, 79, 79, 
	79, 79, 79, 79, 79, 110, 79, 79, 
	79, 79, 79, 79, 79, 79, 79, 79, 
	79, 79, 79, 79, 79, 79, 79, 79, 
	79, 79, 79, 79, 79, 79, 79, 79, 
	79, 141, 79, 79, 79, 79, 79, 79, 
	79, 79, 79, 79, 79, 79, 79, 79, 
	79, 79, 79, 79, 79, 79, 79, 79, 
	79, 79, 79, 79, 79, 79, 173
};

static const int Scanner_start = 16;
static const int Scanner_first_final = 16;
static const int Scanner_error = 0;

static const int Scanner_en_main = 16;


#line 132 "lexer.rl"


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
	GC_disable();
	out = Value::Nil();
	errors = 0;
	lastTokenWasNL = false;
	complete = false;

	pParser = ParseAlloc(GC_malloc);

	/*ParseTrace(trace, 0);*/

	const char *p = data;
	const char *pe = data+len;
	const char* eof = isEof ? pe : 0;
	int cs, act;
	
#line 463 "../parser.cpp"
	{
	cs = Scanner_start;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 202 "lexer.rl"
	
#line 473 "../parser.cpp"
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
#line 494 "../parser.cpp"
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
#line 29 "lexer.rl"
	{act = 1;}
	break;
	case 5:
#line 31 "lexer.rl"
	{act = 3;}
	break;
	case 6:
#line 32 "lexer.rl"
	{act = 4;}
	break;
	case 7:
#line 33 "lexer.rl"
	{act = 5;}
	break;
	case 8:
#line 34 "lexer.rl"
	{act = 6;}
	break;
	case 9:
#line 35 "lexer.rl"
	{act = 7;}
	break;
	case 10:
#line 36 "lexer.rl"
	{act = 8;}
	break;
	case 11:
#line 37 "lexer.rl"
	{act = 9;}
	break;
	case 12:
#line 39 "lexer.rl"
	{act = 10;}
	break;
	case 13:
#line 40 "lexer.rl"
	{act = 11;}
	break;
	case 14:
#line 41 "lexer.rl"
	{act = 12;}
	break;
	case 15:
#line 42 "lexer.rl"
	{act = 13;}
	break;
	case 16:
#line 43 "lexer.rl"
	{act = 14;}
	break;
	case 17:
#line 44 "lexer.rl"
	{act = 15;}
	break;
	case 18:
#line 45 "lexer.rl"
	{act = 16;}
	break;
	case 19:
#line 46 "lexer.rl"
	{act = 17;}
	break;
	case 20:
#line 47 "lexer.rl"
	{act = 18;}
	break;
	case 21:
#line 57 "lexer.rl"
	{act = 21;}
	break;
	case 22:
#line 60 "lexer.rl"
	{act = 22;}
	break;
	case 23:
#line 126 "lexer.rl"
	{act = 66;}
	break;
	case 24:
#line 129 "lexer.rl"
	{act = 67;}
	break;
	case 25:
#line 51 "lexer.rl"
	{te = p+1;{std::string s(ts+1, te-ts-2); token( TOKEN_STR_CONST, Character::c(state.internStr(unescape(s))) );}}
	break;
	case 26:
#line 53 "lexer.rl"
	{te = p+1;{std::string s(ts+1, te-ts-2); token( TOKEN_STR_CONST, Character::c(state.internStr(unescape(s))) );}}
	break;
	case 27:
#line 62 "lexer.rl"
	{te = p+1;{std::string s(ts+1, te-ts-2); token( TOKEN_SYMBOL, CreateSymbol(state.internStr(unescape(s))) );}}
	break;
	case 28:
#line 71 "lexer.rl"
	{te = p+1;{token( TOKEN_NUM_CONST, Integer::c(atof(std::string(ts, te-ts-1).c_str())) );}}
	break;
	case 29:
#line 83 "lexer.rl"
	{te = p+1;{token( TOKEN_PLUS, CreateSymbol(Strings::add) );}}
	break;
	case 30:
#line 85 "lexer.rl"
	{te = p+1;{token( TOKEN_POW, CreateSymbol(Strings::pow) );}}
	break;
	case 31:
#line 86 "lexer.rl"
	{te = p+1;{token( TOKEN_DIVIDE, CreateSymbol(Strings::div) );}}
	break;
	case 32:
#line 88 "lexer.rl"
	{te = p+1;{token( TOKEN_POW, CreateSymbol(Strings::pow) );}}
	break;
	case 33:
#line 89 "lexer.rl"
	{te = p+1;{token( TOKEN_TILDE, CreateSymbol(Strings::tilde) );}}
	break;
	case 34:
#line 90 "lexer.rl"
	{te = p+1;{token( TOKEN_DOLLAR, CreateSymbol(Strings::dollar) );}}
	break;
	case 35:
#line 91 "lexer.rl"
	{te = p+1;{token( TOKEN_AT, CreateSymbol(Strings::at) );}}
	break;
	case 36:
#line 95 "lexer.rl"
	{te = p+1;{token( TOKEN_NS_GET_INT, CreateSymbol(Strings::nsgetint) );}}
	break;
	case 37:
#line 98 "lexer.rl"
	{te = p+1;{token( TOKEN_LBRACE, CreateSymbol(Strings::brace) );}}
	break;
	case 38:
#line 99 "lexer.rl"
	{te = p+1;{token( TOKEN_RBRACE );}}
	break;
	case 39:
#line 100 "lexer.rl"
	{te = p+1;{token( TOKEN_LPAREN, CreateSymbol(Strings::paren) );}}
	break;
	case 40:
#line 101 "lexer.rl"
	{te = p+1;{token( TOKEN_RPAREN );}}
	break;
	case 41:
#line 103 "lexer.rl"
	{te = p+1;{token( TOKEN_LBB, CreateSymbol(Strings::bb) );}}
	break;
	case 42:
#line 104 "lexer.rl"
	{te = p+1;{token( TOKEN_RBRACKET );}}
	break;
	case 43:
#line 107 "lexer.rl"
	{te = p+1;{token( TOKEN_LE, CreateSymbol(Strings::le) );}}
	break;
	case 44:
#line 108 "lexer.rl"
	{te = p+1;{token( TOKEN_GE, CreateSymbol(Strings::ge) );}}
	break;
	case 45:
#line 109 "lexer.rl"
	{te = p+1;{token( TOKEN_EQ, CreateSymbol(Strings::eq) );}}
	break;
	case 46:
#line 110 "lexer.rl"
	{te = p+1;{token( TOKEN_NE, CreateSymbol(Strings::neq) );}}
	break;
	case 47:
#line 111 "lexer.rl"
	{te = p+1;{token( TOKEN_AND2, CreateSymbol(Strings::land2) );}}
	break;
	case 48:
#line 112 "lexer.rl"
	{te = p+1;{token( TOKEN_OR2, CreateSymbol(Strings::lor2) );}}
	break;
	case 49:
#line 113 "lexer.rl"
	{te = p+1;{token( TOKEN_LEFT_ASSIGN, CreateSymbol(Strings::assign) );}}
	break;
	case 50:
#line 115 "lexer.rl"
	{te = p+1;{token( TOKEN_RIGHT_ASSIGN, CreateSymbol(Strings::assign2) );}}
	break;
	case 51:
#line 116 "lexer.rl"
	{te = p+1;{token( TOKEN_LEFT_ASSIGN, CreateSymbol(Strings::assign2) );}}
	break;
	case 52:
#line 117 "lexer.rl"
	{te = p+1;{token( TOKEN_QUESTION, CreateSymbol(Strings::question) );}}
	break;
	case 53:
#line 120 "lexer.rl"
	{te = p+1;{token(TOKEN_SPECIALOP, CreateSymbol(state.internStr(std::string(ts, te-ts))) ); }}
	break;
	case 54:
#line 123 "lexer.rl"
	{te = p+1;{token( TOKEN_COMMA );}}
	break;
	case 55:
#line 124 "lexer.rl"
	{te = p+1;{token( TOKEN_SEMICOLON );}}
	break;
	case 56:
#line 30 "lexer.rl"
	{te = p;p--;{token( TOKEN_NUM_CONST, Logical::NA() );}}
	break;
	case 57:
#line 60 "lexer.rl"
	{te = p;p--;{token( TOKEN_SYMBOL, CreateSymbol(state.internStr(std::string(ts, te-ts))) );}}
	break;
	case 58:
#line 65 "lexer.rl"
	{te = p;p--;{token( TOKEN_NUM_CONST, Double::c(atof(std::string(ts, te-ts).c_str())) );}}
	break;
	case 59:
#line 79 "lexer.rl"
	{te = p;p--;{token( TOKEN_NUM_CONST );}}
	break;
	case 60:
#line 82 "lexer.rl"
	{te = p;p--;{token( TOKEN_EQ_ASSIGN, CreateSymbol(Strings::eqassign) );}}
	break;
	case 61:
#line 84 "lexer.rl"
	{te = p;p--;{token( TOKEN_MINUS, CreateSymbol(Strings::sub) );}}
	break;
	case 62:
#line 87 "lexer.rl"
	{te = p;p--;{token( TOKEN_TIMES, CreateSymbol(Strings::mul) );}}
	break;
	case 63:
#line 92 "lexer.rl"
	{te = p;p--;{token( TOKEN_NOT, CreateSymbol(Strings::lnot) );}}
	break;
	case 64:
#line 93 "lexer.rl"
	{te = p;p--;{token( TOKEN_COLON, CreateSymbol(Strings::colon) );}}
	break;
	case 65:
#line 94 "lexer.rl"
	{te = p;p--;{token( TOKEN_NS_GET, CreateSymbol(Strings::nsget) );}}
	break;
	case 66:
#line 96 "lexer.rl"
	{te = p;p--;{token( TOKEN_AND, CreateSymbol(Strings::land) );}}
	break;
	case 67:
#line 97 "lexer.rl"
	{te = p;p--;{token( TOKEN_OR, CreateSymbol(Strings::lor) );}}
	break;
	case 68:
#line 102 "lexer.rl"
	{te = p;p--;{token( TOKEN_LBRACKET, CreateSymbol(Strings::bracket) );}}
	break;
	case 69:
#line 105 "lexer.rl"
	{te = p;p--;{token( TOKEN_LT, CreateSymbol(Strings::lt) );}}
	break;
	case 70:
#line 106 "lexer.rl"
	{te = p;p--;{token( TOKEN_GT, CreateSymbol(Strings::gt) );}}
	break;
	case 71:
#line 114 "lexer.rl"
	{te = p;p--;{token( TOKEN_RIGHT_ASSIGN, CreateSymbol(Strings::assign) );}}
	break;
	case 72:
#line 126 "lexer.rl"
	{te = p;p--;{token( TOKEN_NEWLINE );}}
	break;
	case 73:
#line 129 "lexer.rl"
	{te = p;p--;}
	break;
	case 74:
#line 65 "lexer.rl"
	{{p = ((te))-1;}{token( TOKEN_NUM_CONST, Double::c(atof(std::string(ts, te-ts).c_str())) );}}
	break;
	case 75:
#line 79 "lexer.rl"
	{{p = ((te))-1;}{token( TOKEN_NUM_CONST );}}
	break;
	case 76:
#line 105 "lexer.rl"
	{{p = ((te))-1;}{token( TOKEN_LT, CreateSymbol(Strings::lt) );}}
	break;
	case 77:
#line 126 "lexer.rl"
	{{p = ((te))-1;}{token( TOKEN_NEWLINE );}}
	break;
	case 78:
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
	{{p = ((te))-1;}std::string s(ts+2, te-ts-2); token( TOKEN_SYMBOL, CreateSymbol((String)-strToInt(s))); }
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
#line 932 "../parser.cpp"
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
#line 949 "../parser.cpp"
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

#line 203 "lexer.rl"
	int syntaxErrors = errors;
	Parse(pParser, 0, Value::Nil(), this);
	ParseFree(pParser, GC_free);
	errors = syntaxErrors;

	if( cs == Scanner_error && syntaxErrors == 0) {
		syntaxErrors++;
		std::cout << "Lexing error (" << intToStr(line+1) << "," << intToStr(col+1) << ") : unexpected '" << std::string(ts, te-ts) + "'" << std::endl; 
	}
	
	GC_enable();
	
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



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
	0, 1, 2, 1, 3, 1, 26, 1, 
	27, 1, 28, 1, 29, 1, 30, 1, 
	31, 1, 32, 1, 33, 1, 34, 1, 
	35, 1, 36, 1, 37, 1, 38, 1, 
	39, 1, 40, 1, 41, 1, 42, 1, 
	43, 1, 44, 1, 45, 1, 46, 1, 
	47, 1, 48, 1, 49, 1, 50, 1, 
	51, 1, 52, 1, 53, 1, 54, 1, 
	55, 1, 56, 1, 57, 1, 58, 1, 
	59, 1, 60, 1, 61, 1, 62, 1, 
	63, 1, 64, 1, 65, 1, 66, 1, 
	67, 1, 68, 1, 69, 1, 70, 1, 
	71, 1, 72, 1, 73, 1, 74, 1, 
	75, 1, 76, 1, 77, 1, 78, 1, 
	79, 1, 80, 2, 0, 1, 2, 3, 
	4, 2, 3, 5, 2, 3, 6, 2, 
	3, 7, 2, 3, 8, 2, 3, 9, 
	2, 3, 10, 2, 3, 11, 2, 3, 
	12, 2, 3, 13, 2, 3, 14, 2, 
	3, 15, 2, 3, 16, 2, 3, 17, 
	2, 3, 18, 2, 3, 19, 2, 3, 
	20, 2, 3, 21, 2, 3, 22, 2, 
	3, 23, 2, 3, 24, 2, 3, 25
	
};

static const short _Scanner_key_offsets[] = {
	0, 0, 4, 5, 8, 8, 10, 13, 
	13, 21, 25, 27, 33, 37, 39, 40, 
	43, 43, 91, 95, 99, 100, 101, 102, 
	103, 104, 112, 120, 126, 130, 138, 145, 
	153, 155, 156, 157, 160, 161, 162, 171, 
	180, 189, 198, 207, 216, 227, 235, 246, 
	256, 265, 274, 283, 292, 301, 310, 319, 
	327, 336, 345, 354, 363, 372, 380, 389, 
	398, 407, 416, 425, 434, 442, 451, 460, 
	469, 477, 486, 495, 504, 513, 522, 531, 
	532, 541, 550, 559, 568, 577, 586, 595, 
	605, 614, 623, 632, 641, 650, 659, 668, 
	678, 687, 696, 705, 714, 723, 732, 741, 
	750, 759, 768, 777, 786
};

static const char _Scanner_trans_keys[] = {
	10, 35, 33, 126, 10, 10, 34, 92, 
	10, 37, 10, 39, 92, 46, 95, 48, 
	57, 65, 90, 97, 122, 43, 45, 48, 
	57, 48, 57, 48, 57, 65, 70, 97, 
	102, 43, 45, 48, 57, 48, 57, 45, 
	10, 92, 96, 10, 33, 34, 35, 36, 
	37, 38, 39, 40, 41, 42, 43, 44, 
	45, 46, 47, 48, 58, 59, 60, 61, 
	62, 63, 64, 70, 73, 78, 84, 91, 
	92, 93, 94, 96, 98, 101, 102, 105, 
	110, 114, 119, 123, 124, 125, 126, 49, 
	57, 65, 122, 10, 35, 33, 126, 10, 
	35, 33, 126, 61, 38, 42, 62, 62, 
	46, 95, 48, 57, 65, 90, 97, 122, 
	46, 95, 48, 57, 65, 90, 97, 122, 
	69, 76, 101, 105, 48, 57, 76, 105, 
	48, 57, 46, 69, 76, 101, 105, 120, 
	48, 57, 46, 69, 76, 101, 105, 48, 
	57, 80, 112, 48, 57, 65, 70, 97, 
	102, 48, 57, 58, 58, 45, 60, 61, 
	61, 61, 46, 65, 95, 48, 57, 66, 
	90, 97, 122, 46, 76, 95, 48, 57, 
	65, 90, 97, 122, 46, 83, 95, 48, 
	57, 65, 90, 97, 122, 46, 69, 95, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	110, 48, 57, 65, 90, 97, 122, 46, 
	95, 102, 48, 57, 65, 90, 97, 122, 
	46, 65, 85, 95, 97, 48, 57, 66, 
	90, 98, 122, 46, 95, 48, 57, 65, 
	90, 97, 122, 46, 95, 99, 105, 114, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	104, 111, 48, 57, 65, 90, 97, 122, 
	46, 95, 97, 48, 57, 65, 90, 98, 
	122, 46, 95, 114, 48, 57, 65, 90, 
	97, 122, 46, 95, 97, 48, 57, 65, 
	90, 98, 122, 46, 95, 99, 48, 57, 
	65, 90, 97, 122, 46, 95, 116, 48, 
	57, 65, 90, 97, 122, 46, 95, 101, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	114, 48, 57, 65, 90, 97, 122, 46, 
	95, 48, 57, 65, 90, 97, 122, 46, 
	95, 109, 48, 57, 65, 90, 97, 122, 
	46, 95, 112, 48, 57, 65, 90, 97, 
	122, 46, 95, 108, 48, 57, 65, 90, 
	97, 122, 46, 95, 101, 48, 57, 65, 
	90, 97, 122, 46, 95, 120, 48, 57, 
	65, 90, 97, 122, 46, 95, 48, 57, 
	65, 90, 97, 122, 46, 95, 110, 48, 
	57, 65, 90, 97, 122, 46, 95, 116, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	101, 48, 57, 65, 90, 97, 122, 46, 
	95, 103, 48, 57, 65, 90, 97, 122, 
	46, 95, 101, 48, 57, 65, 90, 97, 
	122, 46, 95, 114, 48, 57, 65, 90, 
	97, 122, 46, 95, 48, 57, 65, 90, 
	97, 122, 46, 95, 101, 48, 57, 65, 
	90, 97, 122, 46, 95, 97, 48, 57, 
	65, 90, 98, 122, 46, 95, 108, 48, 
	57, 65, 90, 97, 122, 46, 95, 48, 
	57, 65, 90, 97, 122, 46, 76, 95, 
	48, 57, 65, 90, 97, 122, 46, 76, 
	95, 48, 57, 65, 90, 97, 122, 46, 
	78, 95, 48, 57, 65, 90, 97, 122, 
	46, 82, 95, 48, 57, 65, 90, 97, 
	122, 46, 85, 95, 48, 57, 65, 90, 
	97, 122, 46, 69, 95, 48, 57, 65, 
	90, 97, 122, 91, 46, 95, 114, 48, 
	57, 65, 90, 97, 122, 46, 95, 101, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	97, 48, 57, 65, 90, 98, 122, 46, 
	95, 107, 48, 57, 65, 90, 97, 122, 
	46, 95, 108, 48, 57, 65, 90, 97, 
	122, 46, 95, 115, 48, 57, 65, 90, 
	97, 122, 46, 95, 101, 48, 57, 65, 
	90, 97, 122, 46, 95, 111, 117, 48, 
	57, 65, 90, 97, 122, 46, 95, 114, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	110, 48, 57, 65, 90, 97, 122, 46, 
	95, 99, 48, 57, 65, 90, 97, 122, 
	46, 95, 116, 48, 57, 65, 90, 97, 
	122, 46, 95, 105, 48, 57, 65, 90, 
	97, 122, 46, 95, 111, 48, 57, 65, 
	90, 97, 122, 46, 95, 110, 48, 57, 
	65, 90, 97, 122, 46, 95, 102, 110, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	101, 48, 57, 65, 90, 97, 122, 46, 
	95, 120, 48, 57, 65, 90, 97, 122, 
	46, 95, 116, 48, 57, 65, 90, 97, 
	122, 46, 95, 101, 48, 57, 65, 90, 
	97, 122, 46, 95, 112, 48, 57, 65, 
	90, 97, 122, 46, 95, 101, 48, 57, 
	65, 90, 97, 122, 46, 95, 97, 48, 
	57, 65, 90, 98, 122, 46, 95, 116, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	104, 48, 57, 65, 90, 97, 122, 46, 
	95, 105, 48, 57, 65, 90, 97, 122, 
	46, 95, 108, 48, 57, 65, 90, 97, 
	122, 46, 95, 101, 48, 57, 65, 90, 
	97, 122, 124, 0
};

static const char _Scanner_single_lengths[] = {
	0, 2, 1, 3, 0, 2, 3, 0, 
	2, 2, 0, 0, 2, 0, 1, 3, 
	0, 44, 2, 2, 1, 1, 1, 1, 
	1, 2, 2, 4, 2, 6, 5, 2, 
	0, 1, 1, 3, 1, 1, 3, 3, 
	3, 3, 3, 3, 5, 2, 5, 4, 
	3, 3, 3, 3, 3, 3, 3, 2, 
	3, 3, 3, 3, 3, 2, 3, 3, 
	3, 3, 3, 3, 2, 3, 3, 3, 
	2, 3, 3, 3, 3, 3, 3, 1, 
	3, 3, 3, 3, 3, 3, 3, 4, 
	3, 3, 3, 3, 3, 3, 3, 4, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 1
};

static const char _Scanner_range_lengths[] = {
	0, 1, 0, 0, 0, 0, 0, 0, 
	3, 1, 1, 3, 1, 1, 0, 0, 
	0, 2, 1, 1, 0, 0, 0, 0, 
	0, 3, 3, 1, 1, 1, 1, 3, 
	1, 0, 0, 0, 0, 0, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 0, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 0
};

static const short _Scanner_index_offsets[] = {
	0, 0, 4, 6, 10, 11, 14, 18, 
	19, 25, 29, 31, 35, 39, 41, 43, 
	47, 48, 95, 99, 103, 105, 107, 109, 
	111, 113, 119, 125, 131, 135, 143, 150, 
	156, 158, 160, 162, 166, 168, 170, 177, 
	184, 191, 198, 205, 212, 221, 227, 236, 
	244, 251, 258, 265, 272, 279, 286, 293, 
	299, 306, 313, 320, 327, 334, 340, 347, 
	354, 361, 368, 375, 382, 388, 395, 402, 
	409, 415, 422, 429, 436, 443, 450, 457, 
	459, 466, 473, 480, 487, 494, 501, 508, 
	516, 523, 530, 537, 544, 551, 558, 565, 
	573, 580, 587, 594, 601, 608, 615, 622, 
	629, 636, 643, 650, 657
};

static const unsigned char _Scanner_indicies[] = {
	2, 3, 0, 1, 2, 3, 6, 7, 
	8, 5, 5, 6, 10, 9, 6, 12, 
	13, 11, 11, 14, 16, 15, 16, 16, 
	6, 18, 18, 19, 17, 19, 17, 20, 
	20, 20, 17, 22, 22, 23, 21, 23, 
	21, 25, 24, 6, 27, 28, 26, 26, 
	2, 30, 5, 3, 31, 9, 32, 11, 
	33, 34, 35, 36, 37, 38, 39, 40, 
	41, 43, 44, 45, 46, 47, 48, 49, 
	50, 51, 52, 53, 54, 6, 55, 56, 
	26, 57, 58, 59, 60, 61, 62, 63, 
	64, 65, 66, 67, 42, 16, 29, 2, 
	3, 68, 29, 2, 3, 69, 1, 71, 
	70, 73, 72, 75, 74, 77, 76, 79, 
	78, 16, 16, 80, 16, 16, 4, 16, 
	16, 16, 16, 16, 4, 82, 83, 82, 
	84, 15, 81, 83, 84, 19, 81, 15, 
	82, 83, 82, 84, 85, 42, 81, 15, 
	82, 83, 82, 84, 42, 81, 87, 87, 
	20, 20, 20, 86, 23, 86, 89, 88, 
	91, 90, 93, 94, 95, 92, 97, 96, 
	99, 98, 16, 101, 16, 16, 16, 16, 
	100, 16, 102, 16, 16, 16, 16, 100, 
	16, 103, 16, 16, 16, 16, 100, 16, 
	104, 16, 16, 16, 16, 100, 16, 16, 
	105, 16, 16, 16, 100, 16, 16, 106, 
	16, 16, 16, 100, 16, 107, 108, 16, 
	109, 16, 16, 16, 100, 16, 111, 16, 
	16, 16, 110, 16, 16, 112, 113, 114, 
	16, 16, 16, 100, 16, 16, 115, 116, 
	16, 16, 16, 100, 16, 16, 117, 16, 
	16, 16, 100, 16, 16, 118, 16, 16, 
	16, 100, 16, 16, 119, 16, 16, 16, 
	100, 16, 16, 120, 16, 16, 16, 100, 
	16, 16, 121, 16, 16, 16, 100, 16, 
	16, 122, 16, 16, 16, 100, 16, 16, 
	123, 16, 16, 16, 100, 16, 124, 16, 
	16, 16, 100, 16, 16, 125, 16, 16, 
	16, 100, 16, 16, 126, 16, 16, 16, 
	100, 16, 16, 127, 16, 16, 16, 100, 
	16, 16, 128, 16, 16, 16, 100, 16, 
	16, 129, 16, 16, 16, 100, 16, 130, 
	16, 16, 16, 100, 16, 16, 131, 16, 
	16, 16, 100, 16, 16, 132, 16, 16, 
	16, 100, 16, 16, 133, 16, 16, 16, 
	100, 16, 16, 134, 16, 16, 16, 100, 
	16, 16, 135, 16, 16, 16, 100, 16, 
	16, 136, 16, 16, 16, 100, 16, 137, 
	16, 16, 16, 100, 16, 16, 138, 16, 
	16, 16, 100, 16, 16, 139, 16, 16, 
	16, 100, 16, 16, 140, 16, 16, 16, 
	100, 16, 141, 16, 16, 16, 100, 16, 
	142, 16, 16, 16, 16, 100, 16, 143, 
	16, 16, 16, 16, 100, 16, 144, 16, 
	16, 16, 16, 100, 16, 145, 16, 16, 
	16, 16, 100, 16, 146, 16, 16, 16, 
	16, 100, 16, 147, 16, 16, 16, 16, 
	100, 149, 148, 16, 16, 150, 16, 16, 
	16, 100, 16, 16, 151, 16, 16, 16, 
	100, 16, 16, 152, 16, 16, 16, 100, 
	16, 16, 153, 16, 16, 16, 100, 16, 
	16, 154, 16, 16, 16, 100, 16, 16, 
	155, 16, 16, 16, 100, 16, 16, 156, 
	16, 16, 16, 100, 16, 16, 157, 158, 
	16, 16, 16, 100, 16, 16, 159, 16, 
	16, 16, 100, 16, 16, 160, 16, 16, 
	16, 100, 16, 16, 161, 16, 16, 16, 
	100, 16, 16, 162, 16, 16, 16, 100, 
	16, 16, 163, 16, 16, 16, 100, 16, 
	16, 164, 16, 16, 16, 100, 16, 16, 
	165, 16, 16, 16, 100, 16, 16, 166, 
	167, 16, 16, 16, 100, 16, 16, 168, 
	16, 16, 16, 100, 16, 16, 169, 16, 
	16, 16, 100, 16, 16, 170, 16, 16, 
	16, 100, 16, 16, 171, 16, 16, 16, 
	100, 16, 16, 172, 16, 16, 16, 100, 
	16, 16, 173, 16, 16, 16, 100, 16, 
	16, 174, 16, 16, 16, 100, 16, 16, 
	175, 16, 16, 16, 100, 16, 16, 176, 
	16, 16, 16, 100, 16, 16, 177, 16, 
	16, 16, 100, 16, 16, 178, 16, 16, 
	16, 100, 16, 16, 179, 16, 16, 16, 
	100, 181, 180, 0
};

static const char _Scanner_trans_targs[] = {
	17, 1, 19, 2, 17, 3, 0, 17, 
	4, 5, 17, 6, 17, 7, 25, 27, 
	26, 17, 10, 28, 31, 17, 13, 32, 
	17, 17, 15, 16, 17, 18, 20, 17, 
	21, 17, 17, 22, 17, 17, 23, 8, 
	17, 29, 30, 33, 17, 35, 36, 37, 
	17, 17, 38, 42, 44, 76, 79, 17, 
	17, 80, 84, 87, 95, 96, 99, 104, 
	17, 108, 17, 17, 17, 17, 17, 17, 
	17, 17, 17, 17, 17, 24, 17, 17, 
	25, 17, 9, 17, 17, 11, 17, 12, 
	17, 34, 17, 17, 17, 17, 14, 17, 
	17, 17, 17, 17, 17, 39, 40, 41, 
	26, 43, 26, 45, 73, 75, 17, 46, 
	47, 62, 69, 48, 56, 49, 50, 51, 
	52, 53, 54, 55, 26, 57, 58, 59, 
	60, 61, 26, 63, 64, 65, 66, 67, 
	68, 26, 70, 71, 72, 26, 74, 26, 
	26, 77, 78, 26, 17, 17, 81, 82, 
	83, 26, 85, 86, 26, 88, 89, 26, 
	90, 91, 92, 93, 94, 26, 26, 26, 
	97, 98, 26, 100, 101, 102, 103, 26, 
	105, 106, 107, 26, 17, 17
};

static const unsigned char _Scanner_trans_actions[] = {
	111, 0, 178, 0, 113, 0, 0, 7, 
	0, 0, 63, 0, 5, 0, 175, 3, 
	175, 105, 0, 0, 3, 107, 0, 0, 
	109, 59, 0, 0, 9, 181, 0, 25, 
	0, 35, 37, 0, 15, 65, 0, 0, 
	19, 3, 3, 0, 67, 3, 0, 0, 
	61, 27, 0, 0, 0, 0, 0, 41, 
	17, 0, 0, 0, 0, 0, 0, 0, 
	31, 0, 33, 23, 103, 101, 83, 49, 
	89, 51, 81, 21, 79, 0, 99, 57, 
	172, 73, 0, 13, 11, 0, 75, 0, 
	85, 0, 87, 29, 95, 55, 0, 43, 
	77, 47, 97, 45, 71, 0, 0, 0, 
	124, 0, 127, 0, 0, 0, 69, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 139, 0, 0, 0, 
	0, 0, 142, 0, 0, 0, 0, 0, 
	0, 133, 0, 0, 0, 136, 0, 118, 
	130, 0, 0, 121, 93, 39, 0, 0, 
	0, 169, 0, 0, 163, 0, 0, 154, 
	0, 0, 0, 0, 0, 145, 157, 160, 
	0, 0, 166, 0, 0, 0, 0, 151, 
	0, 0, 0, 148, 91, 53
};

static const unsigned char _Scanner_to_state_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 115, 0, 0, 0, 0, 0, 0, 
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
	0, 0, 0, 0, 0
};

static const unsigned char _Scanner_from_state_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 1, 0, 0, 0, 0, 0, 0, 
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
	0, 0, 0, 0, 0
};

static const short _Scanner_eof_trans[] = {
	0, 1, 5, 0, 0, 0, 0, 0, 
	0, 18, 18, 18, 22, 22, 25, 0, 
	0, 0, 69, 70, 71, 73, 75, 77, 
	79, 5, 5, 82, 82, 82, 82, 87, 
	87, 89, 91, 93, 97, 99, 101, 101, 
	101, 101, 101, 101, 101, 111, 101, 101, 
	101, 101, 101, 101, 101, 101, 101, 101, 
	101, 101, 101, 101, 101, 101, 101, 101, 
	101, 101, 101, 101, 101, 101, 101, 101, 
	101, 101, 101, 101, 101, 101, 101, 149, 
	101, 101, 101, 101, 101, 101, 101, 101, 
	101, 101, 101, 101, 101, 101, 101, 101, 
	101, 101, 101, 101, 101, 101, 101, 101, 
	101, 101, 101, 101, 181
};

static const int Scanner_start = 17;
static const int Scanner_first_final = 17;
static const int Scanner_error = 0;

static const int Scanner_en_main = 17;


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
	
#line 485 "../parser.cpp"
	{
	cs = Scanner_start;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 201 "lexer.rl"
	
#line 495 "../parser.cpp"
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
#line 516 "../parser.cpp"
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
#line 38 "lexer.rl"
	{act = 10;}
	break;
	case 13:
#line 39 "lexer.rl"
	{act = 11;}
	break;
	case 14:
#line 40 "lexer.rl"
	{act = 12;}
	break;
	case 15:
#line 41 "lexer.rl"
	{act = 13;}
	break;
	case 16:
#line 42 "lexer.rl"
	{act = 14;}
	break;
	case 17:
#line 43 "lexer.rl"
	{act = 15;}
	break;
	case 18:
#line 44 "lexer.rl"
	{act = 16;}
	break;
	case 19:
#line 45 "lexer.rl"
	{act = 17;}
	break;
	case 20:
#line 46 "lexer.rl"
	{act = 18;}
	break;
	case 21:
#line 47 "lexer.rl"
	{act = 19;}
	break;
	case 22:
#line 57 "lexer.rl"
	{act = 22;}
	break;
	case 23:
#line 60 "lexer.rl"
	{act = 23;}
	break;
	case 24:
#line 126 "lexer.rl"
	{act = 68;}
	break;
	case 25:
#line 129 "lexer.rl"
	{act = 69;}
	break;
	case 26:
#line 51 "lexer.rl"
	{te = p+1;{std::string s(ts+1, te-ts-2); token( TOKEN_STR_CONST, Character::c(state.StrToSym(unescape(s))) );}}
	break;
	case 27:
#line 53 "lexer.rl"
	{te = p+1;{std::string s(ts+1, te-ts-2); token( TOKEN_STR_CONST, Character::c(state.StrToSym(unescape(s))) );}}
	break;
	case 28:
#line 62 "lexer.rl"
	{te = p+1;{std::string s(ts+1, te-ts-2); token( TOKEN_SYMBOL, Symbol(state.StrToSym(unescape(s))) );}}
	break;
	case 29:
#line 68 "lexer.rl"
	{te = p+1;{token( TOKEN_NUM_CONST, Complex::c(std::complex<double>(0, atof(std::string(ts, te-ts-1).c_str()))) );}}
	break;
	case 30:
#line 71 "lexer.rl"
	{te = p+1;{token( TOKEN_NUM_CONST, Integer::c(atof(std::string(ts, te-ts-1).c_str())) );}}
	break;
	case 31:
#line 83 "lexer.rl"
	{te = p+1;{token( TOKEN_PLUS, Symbols::add );}}
	break;
	case 32:
#line 85 "lexer.rl"
	{te = p+1;{token( TOKEN_POW, Symbols::pow );}}
	break;
	case 33:
#line 86 "lexer.rl"
	{te = p+1;{token( TOKEN_DIVIDE, Symbols::div );}}
	break;
	case 34:
#line 88 "lexer.rl"
	{te = p+1;{token( TOKEN_POW, Symbols::pow );}}
	break;
	case 35:
#line 89 "lexer.rl"
	{te = p+1;{token( TOKEN_TILDE, Symbols::tilde );}}
	break;
	case 36:
#line 90 "lexer.rl"
	{te = p+1;{token( TOKEN_DOLLAR, Symbols::dollar );}}
	break;
	case 37:
#line 91 "lexer.rl"
	{te = p+1;{token( TOKEN_AT, Symbols::at );}}
	break;
	case 38:
#line 95 "lexer.rl"
	{te = p+1;{token( TOKEN_NS_GET_INT, Symbols::nsgetint );}}
	break;
	case 39:
#line 98 "lexer.rl"
	{te = p+1;{token( TOKEN_LBRACE, Symbols::brace );}}
	break;
	case 40:
#line 99 "lexer.rl"
	{te = p+1;{token( TOKEN_RBRACE );}}
	break;
	case 41:
#line 100 "lexer.rl"
	{te = p+1;{token( TOKEN_LPAREN, Symbols::paren );}}
	break;
	case 42:
#line 101 "lexer.rl"
	{te = p+1;{token( TOKEN_RPAREN );}}
	break;
	case 43:
#line 103 "lexer.rl"
	{te = p+1;{token( TOKEN_LBB, Symbols::bb );}}
	break;
	case 44:
#line 104 "lexer.rl"
	{te = p+1;{token( TOKEN_RBRACKET );}}
	break;
	case 45:
#line 107 "lexer.rl"
	{te = p+1;{token( TOKEN_LE, Symbols::le );}}
	break;
	case 46:
#line 108 "lexer.rl"
	{te = p+1;{token( TOKEN_GE, Symbols::ge );}}
	break;
	case 47:
#line 109 "lexer.rl"
	{te = p+1;{token( TOKEN_EQ, Symbols::eq );}}
	break;
	case 48:
#line 110 "lexer.rl"
	{te = p+1;{token( TOKEN_NE, Symbols::neq );}}
	break;
	case 49:
#line 111 "lexer.rl"
	{te = p+1;{token( TOKEN_AND2, Symbols::sland );}}
	break;
	case 50:
#line 112 "lexer.rl"
	{te = p+1;{token( TOKEN_OR2, Symbols::slor );}}
	break;
	case 51:
#line 113 "lexer.rl"
	{te = p+1;{token( TOKEN_LEFT_ASSIGN, Symbols::assign );}}
	break;
	case 52:
#line 115 "lexer.rl"
	{te = p+1;{token( TOKEN_RIGHT_ASSIGN, Symbols::assign2 );}}
	break;
	case 53:
#line 116 "lexer.rl"
	{te = p+1;{token( TOKEN_LEFT_ASSIGN, Symbols::assign2 );}}
	break;
	case 54:
#line 117 "lexer.rl"
	{te = p+1;{token( TOKEN_QUESTION, Symbols::question );}}
	break;
	case 55:
#line 120 "lexer.rl"
	{te = p+1;{token(TOKEN_SPECIALOP, Symbol(state.StrToSym(std::string(ts, te-ts))) ); }}
	break;
	case 56:
#line 123 "lexer.rl"
	{te = p+1;{token( TOKEN_COMMA );}}
	break;
	case 57:
#line 124 "lexer.rl"
	{te = p+1;{token( TOKEN_SEMICOLON );}}
	break;
	case 58:
#line 30 "lexer.rl"
	{te = p;p--;{token( TOKEN_NUM_CONST, Logical::NA() );}}
	break;
	case 59:
#line 60 "lexer.rl"
	{te = p;p--;{token( TOKEN_SYMBOL, Symbol(state.StrToSym(std::string(ts, te-ts))) );}}
	break;
	case 60:
#line 65 "lexer.rl"
	{te = p;p--;{token( TOKEN_NUM_CONST, Double::c(atof(std::string(ts, te-ts).c_str())) );}}
	break;
	case 61:
#line 79 "lexer.rl"
	{te = p;p--;{token( TOKEN_NUM_CONST );}}
	break;
	case 62:
#line 82 "lexer.rl"
	{te = p;p--;{token( TOKEN_EQ_ASSIGN, Symbols::eqassign );}}
	break;
	case 63:
#line 84 "lexer.rl"
	{te = p;p--;{token( TOKEN_MINUS, Symbols::sub );}}
	break;
	case 64:
#line 87 "lexer.rl"
	{te = p;p--;{token( TOKEN_TIMES, Symbols::mul );}}
	break;
	case 65:
#line 92 "lexer.rl"
	{te = p;p--;{token( TOKEN_NOT, Symbols::lnot );}}
	break;
	case 66:
#line 93 "lexer.rl"
	{te = p;p--;{token( TOKEN_COLON, Symbols::colon );}}
	break;
	case 67:
#line 94 "lexer.rl"
	{te = p;p--;{token( TOKEN_NS_GET, Symbols::nsget );}}
	break;
	case 68:
#line 96 "lexer.rl"
	{te = p;p--;{token( TOKEN_AND, Symbols::land );}}
	break;
	case 69:
#line 97 "lexer.rl"
	{te = p;p--;{token( TOKEN_OR, Symbols::lor );}}
	break;
	case 70:
#line 102 "lexer.rl"
	{te = p;p--;{token( TOKEN_LBRACKET, Symbols::bracket );}}
	break;
	case 71:
#line 105 "lexer.rl"
	{te = p;p--;{token( TOKEN_LT, Symbols::lt );}}
	break;
	case 72:
#line 106 "lexer.rl"
	{te = p;p--;{token( TOKEN_GT, Symbols::gt );}}
	break;
	case 73:
#line 114 "lexer.rl"
	{te = p;p--;{token( TOKEN_RIGHT_ASSIGN, Symbols::assign );}}
	break;
	case 74:
#line 126 "lexer.rl"
	{te = p;p--;{token( TOKEN_NEWLINE );}}
	break;
	case 75:
#line 129 "lexer.rl"
	{te = p;p--;}
	break;
	case 76:
#line 65 "lexer.rl"
	{{p = ((te))-1;}{token( TOKEN_NUM_CONST, Double::c(atof(std::string(ts, te-ts).c_str())) );}}
	break;
	case 77:
#line 79 "lexer.rl"
	{{p = ((te))-1;}{token( TOKEN_NUM_CONST );}}
	break;
	case 78:
#line 105 "lexer.rl"
	{{p = ((te))-1;}{token( TOKEN_LT, Symbols::lt );}}
	break;
	case 79:
#line 126 "lexer.rl"
	{{p = ((te))-1;}{token( TOKEN_NEWLINE );}}
	break;
	case 80:
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
	{{p = ((te))-1;}token( TOKEN_NUM_CONST, Complex::NA() );}
	break;
	case 11:
	{{p = ((te))-1;}token( TOKEN_FUNCTION, Symbols::function );}
	break;
	case 12:
	{{p = ((te))-1;}token( TOKEN_WHILE, Symbols::whileSym );}
	break;
	case 13:
	{{p = ((te))-1;}token( TOKEN_REPEAT, Symbols::repeatSym );}
	break;
	case 14:
	{{p = ((te))-1;}token( TOKEN_FOR, Symbols::forSym );}
	break;
	case 15:
	{{p = ((te))-1;}token( TOKEN_IF, Symbols::ifSym );}
	break;
	case 16:
	{{p = ((te))-1;}token( TOKEN_IN );}
	break;
	case 17:
	{{p = ((te))-1;}token( TOKEN_ELSE );}
	break;
	case 18:
	{{p = ((te))-1;}token( TOKEN_NEXT, Symbols::nextSym );}
	break;
	case 19:
	{{p = ((te))-1;}token( TOKEN_BREAK, Symbols::breakSym );}
	break;
	case 22:
	{{p = ((te))-1;}std::string s(ts+2, te-ts-2); token( TOKEN_SYMBOL, Symbol(-strToInt(s)));}
	break;
	case 23:
	{{p = ((te))-1;}token( TOKEN_SYMBOL, Symbol(state.StrToSym(std::string(ts, te-ts))) );}
	break;
	case 68:
	{{p = ((te))-1;}token( TOKEN_NEWLINE );}
	break;
	default:
	{{p = ((te))-1;}}
	break;
	}
	}
	break;
#line 965 "../parser.cpp"
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
#line 982 "../parser.cpp"
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

#line 202 "lexer.rl"
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


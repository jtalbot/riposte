
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
#include "../interpreter.h"
#include "grammar.h"
#include "grammar.cpp"


#line 22 "lexer.cpp"
static const char _Scanner_actions[] = {
	0, 1, 2, 1, 3, 1, 27, 1, 
	28, 1, 29, 1, 30, 1, 31, 1, 
	32, 1, 33, 1, 34, 1, 35, 1, 
	36, 1, 37, 1, 38, 1, 39, 1, 
	40, 1, 41, 1, 42, 1, 43, 1, 
	44, 1, 45, 1, 46, 1, 47, 1, 
	48, 1, 49, 1, 50, 1, 51, 1, 
	52, 1, 53, 1, 54, 1, 55, 1, 
	56, 1, 57, 1, 58, 1, 59, 1, 
	60, 1, 61, 1, 62, 1, 63, 1, 
	64, 1, 65, 1, 66, 1, 67, 1, 
	68, 1, 69, 1, 70, 1, 71, 1, 
	72, 1, 73, 1, 74, 1, 75, 1, 
	76, 1, 77, 1, 78, 1, 79, 1, 
	80, 1, 81, 1, 82, 1, 83, 1, 
	84, 2, 0, 1, 2, 3, 4, 2, 
	3, 5, 2, 3, 6, 2, 3, 7, 
	2, 3, 8, 2, 3, 9, 2, 3, 
	10, 2, 3, 11, 2, 3, 12, 2, 
	3, 13, 2, 3, 14, 2, 3, 15, 
	2, 3, 16, 2, 3, 17, 2, 3, 
	18, 2, 3, 19, 2, 3, 20, 2, 
	3, 21, 2, 3, 22, 2, 3, 23, 
	2, 3, 24, 2, 3, 25, 2, 3, 
	26
};

static const short _Scanner_key_offsets[] = {
	0, 0, 4, 5, 7, 7, 9, 11, 
	11, 15, 17, 23, 27, 29, 30, 32, 
	32, 81, 85, 89, 90, 91, 92, 93, 
	94, 95, 103, 111, 119, 125, 129, 137, 
	144, 153, 156, 157, 158, 161, 162, 163, 
	164, 173, 182, 191, 200, 209, 218, 230, 
	238, 250, 259, 268, 277, 286, 295, 304, 
	313, 322, 330, 339, 348, 357, 366, 375, 
	384, 392, 401, 410, 419, 427, 436, 445, 
	454, 462, 471, 480, 489, 498, 507, 516, 
	525, 526, 535, 544, 553, 562, 571, 580, 
	589, 599, 608, 617, 626, 635, 644, 653, 
	662, 672, 681, 690, 699, 708, 717, 726, 
	735, 744, 753, 762, 771, 780
};

static const char _Scanner_trans_keys[] = {
	10, 35, 33, 126, 10, 34, 92, 10, 
	37, 39, 92, 43, 45, 48, 57, 48, 
	57, 48, 57, 65, 70, 97, 102, 43, 
	45, 48, 57, 48, 57, 45, 92, 96, 
	10, 33, 34, 35, 36, 37, 38, 39, 
	40, 41, 42, 43, 44, 45, 46, 47, 
	48, 58, 59, 60, 61, 62, 63, 64, 
	70, 73, 78, 84, 91, 92, 93, 94, 
	95, 96, 98, 101, 102, 105, 110, 114, 
	119, 123, 124, 125, 126, 49, 57, 65, 
	122, 10, 35, 33, 126, 10, 35, 33, 
	126, 61, 61, 38, 42, 62, 62, 46, 
	95, 48, 57, 65, 90, 97, 122, 46, 
	95, 48, 57, 65, 90, 97, 122, 46, 
	95, 48, 57, 65, 90, 97, 122, 69, 
	76, 101, 105, 48, 57, 76, 105, 48, 
	57, 46, 69, 76, 101, 105, 120, 48, 
	57, 46, 69, 76, 101, 105, 48, 57, 
	76, 80, 112, 48, 57, 65, 70, 97, 
	102, 76, 48, 57, 58, 58, 45, 60, 
	61, 61, 61, 61, 46, 65, 95, 48, 
	57, 66, 90, 97, 122, 46, 76, 95, 
	48, 57, 65, 90, 97, 122, 46, 83, 
	95, 48, 57, 65, 90, 97, 122, 46, 
	69, 95, 48, 57, 65, 90, 97, 122, 
	46, 95, 110, 48, 57, 65, 90, 97, 
	122, 46, 95, 102, 48, 57, 65, 90, 
	97, 122, 46, 65, 85, 95, 97, 105, 
	48, 57, 66, 90, 98, 122, 46, 95, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	99, 105, 108, 114, 48, 57, 65, 90, 
	97, 122, 46, 95, 104, 48, 57, 65, 
	90, 97, 122, 46, 95, 97, 48, 57, 
	65, 90, 98, 122, 46, 95, 114, 48, 
	57, 65, 90, 97, 122, 46, 95, 97, 
	48, 57, 65, 90, 98, 122, 46, 95, 
	99, 48, 57, 65, 90, 97, 122, 46, 
	95, 116, 48, 57, 65, 90, 97, 122, 
	46, 95, 101, 48, 57, 65, 90, 97, 
	122, 46, 95, 114, 48, 57, 65, 90, 
	97, 122, 46, 95, 48, 57, 65, 90, 
	97, 122, 46, 95, 110, 48, 57, 65, 
	90, 97, 122, 46, 95, 116, 48, 57, 
	65, 90, 97, 122, 46, 95, 101, 48, 
	57, 65, 90, 97, 122, 46, 95, 103, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	101, 48, 57, 65, 90, 97, 122, 46, 
	95, 114, 48, 57, 65, 90, 97, 122, 
	46, 95, 48, 57, 65, 90, 97, 122, 
	46, 95, 105, 48, 57, 65, 90, 97, 
	122, 46, 95, 115, 48, 57, 65, 90, 
	97, 122, 46, 95, 116, 48, 57, 65, 
	90, 97, 122, 46, 95, 48, 57, 65, 
	90, 97, 122, 46, 95, 101, 48, 57, 
	65, 90, 97, 122, 46, 95, 97, 48, 
	57, 65, 90, 98, 122, 46, 95, 108, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	48, 57, 65, 90, 97, 122, 46, 76, 
	95, 48, 57, 65, 90, 97, 122, 46, 
	76, 95, 48, 57, 65, 90, 97, 122, 
	46, 78, 95, 48, 57, 65, 90, 97, 
	122, 46, 95, 108, 48, 57, 65, 90, 
	97, 122, 46, 82, 95, 48, 57, 65, 
	90, 97, 122, 46, 85, 95, 48, 57, 
	65, 90, 97, 122, 46, 69, 95, 48, 
	57, 65, 90, 97, 122, 91, 46, 95, 
	114, 48, 57, 65, 90, 97, 122, 46, 
	95, 101, 48, 57, 65, 90, 97, 122, 
	46, 95, 97, 48, 57, 65, 90, 98, 
	122, 46, 95, 107, 48, 57, 65, 90, 
	97, 122, 46, 95, 108, 48, 57, 65, 
	90, 97, 122, 46, 95, 115, 48, 57, 
	65, 90, 97, 122, 46, 95, 101, 48, 
	57, 65, 90, 97, 122, 46, 95, 111, 
	117, 48, 57, 65, 90, 97, 122, 46, 
	95, 114, 48, 57, 65, 90, 97, 122, 
	46, 95, 110, 48, 57, 65, 90, 97, 
	122, 46, 95, 99, 48, 57, 65, 90, 
	97, 122, 46, 95, 116, 48, 57, 65, 
	90, 97, 122, 46, 95, 105, 48, 57, 
	65, 90, 97, 122, 46, 95, 111, 48, 
	57, 65, 90, 97, 122, 46, 95, 110, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	102, 110, 48, 57, 65, 90, 97, 122, 
	46, 95, 101, 48, 57, 65, 90, 97, 
	122, 46, 95, 120, 48, 57, 65, 90, 
	97, 122, 46, 95, 116, 48, 57, 65, 
	90, 97, 122, 46, 95, 101, 48, 57, 
	65, 90, 97, 122, 46, 95, 112, 48, 
	57, 65, 90, 97, 122, 46, 95, 101, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	97, 48, 57, 65, 90, 98, 122, 46, 
	95, 116, 48, 57, 65, 90, 97, 122, 
	46, 95, 104, 48, 57, 65, 90, 97, 
	122, 46, 95, 105, 48, 57, 65, 90, 
	97, 122, 46, 95, 108, 48, 57, 65, 
	90, 97, 122, 46, 95, 101, 48, 57, 
	65, 90, 97, 122, 124, 0
};

static const char _Scanner_single_lengths[] = {
	0, 2, 1, 2, 0, 2, 2, 0, 
	2, 0, 0, 2, 0, 1, 2, 0, 
	45, 2, 2, 1, 1, 1, 1, 1, 
	1, 2, 2, 2, 4, 2, 6, 5, 
	3, 1, 1, 1, 3, 1, 1, 1, 
	3, 3, 3, 3, 3, 3, 6, 2, 
	6, 3, 3, 3, 3, 3, 3, 3, 
	3, 2, 3, 3, 3, 3, 3, 3, 
	2, 3, 3, 3, 2, 3, 3, 3, 
	2, 3, 3, 3, 3, 3, 3, 3, 
	1, 3, 3, 3, 3, 3, 3, 3, 
	4, 3, 3, 3, 3, 3, 3, 3, 
	4, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 1
};

static const char _Scanner_range_lengths[] = {
	0, 1, 0, 0, 0, 0, 0, 0, 
	1, 1, 3, 1, 1, 0, 0, 0, 
	2, 1, 1, 0, 0, 0, 0, 0, 
	0, 3, 3, 3, 1, 1, 1, 1, 
	3, 1, 0, 0, 0, 0, 0, 0, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	0, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 0
};

static const short _Scanner_index_offsets[] = {
	0, 0, 4, 6, 9, 10, 13, 16, 
	17, 21, 23, 27, 31, 33, 35, 38, 
	39, 87, 91, 95, 97, 99, 101, 103, 
	105, 107, 113, 119, 125, 131, 135, 143, 
	150, 157, 160, 162, 164, 168, 170, 172, 
	174, 181, 188, 195, 202, 209, 216, 226, 
	232, 242, 249, 256, 263, 270, 277, 284, 
	291, 298, 304, 311, 318, 325, 332, 339, 
	346, 352, 359, 366, 373, 379, 386, 393, 
	400, 406, 413, 420, 427, 434, 441, 448, 
	455, 457, 464, 471, 478, 485, 492, 499, 
	506, 514, 521, 528, 535, 542, 549, 556, 
	563, 571, 578, 585, 592, 599, 606, 613, 
	620, 627, 634, 641, 648, 655
};

static const unsigned char _Scanner_indicies[] = {
	2, 3, 0, 1, 2, 3, 6, 7, 
	5, 5, 9, 10, 8, 12, 13, 11, 
	11, 15, 15, 16, 14, 16, 14, 17, 
	17, 17, 14, 19, 19, 20, 18, 20, 
	18, 22, 21, 24, 25, 23, 23, 2, 
	27, 5, 3, 28, 8, 29, 11, 30, 
	31, 32, 33, 34, 35, 36, 37, 38, 
	40, 41, 42, 43, 44, 45, 46, 48, 
	49, 50, 51, 52, 9, 53, 54, 9, 
	23, 55, 56, 57, 58, 59, 60, 61, 
	62, 63, 64, 65, 39, 47, 26, 2, 
	3, 66, 26, 2, 3, 67, 1, 69, 
	68, 71, 70, 73, 72, 75, 74, 77, 
	76, 79, 78, 81, 47, 82, 47, 47, 
	80, 47, 47, 83, 47, 47, 4, 47, 
	47, 47, 47, 47, 4, 85, 86, 85, 
	87, 82, 84, 86, 87, 16, 84, 82, 
	85, 86, 85, 87, 88, 39, 84, 82, 
	85, 86, 85, 87, 39, 84, 90, 91, 
	91, 17, 17, 17, 89, 90, 20, 89, 
	93, 92, 95, 94, 97, 98, 99, 96, 
	101, 100, 103, 102, 105, 104, 47, 106, 
	47, 47, 47, 47, 80, 47, 107, 47, 
	47, 47, 47, 80, 47, 108, 47, 47, 
	47, 47, 80, 47, 109, 47, 47, 47, 
	47, 80, 47, 47, 110, 47, 47, 47, 
	80, 47, 47, 111, 47, 47, 47, 80, 
	47, 112, 113, 47, 114, 115, 47, 47, 
	47, 80, 47, 117, 47, 47, 47, 116, 
	47, 47, 118, 119, 120, 121, 47, 47, 
	47, 80, 47, 47, 122, 47, 47, 47, 
	80, 47, 47, 123, 47, 47, 47, 80, 
	47, 47, 124, 47, 47, 47, 80, 47, 
	47, 125, 47, 47, 47, 80, 47, 47, 
	126, 47, 47, 47, 80, 47, 47, 127, 
	47, 47, 47, 80, 47, 47, 128, 47, 
	47, 47, 80, 47, 47, 129, 47, 47, 
	47, 80, 47, 130, 47, 47, 47, 80, 
	47, 47, 131, 47, 47, 47, 80, 47, 
	47, 132, 47, 47, 47, 80, 47, 47, 
	133, 47, 47, 47, 80, 47, 47, 134, 
	47, 47, 47, 80, 47, 47, 135, 47, 
	47, 47, 80, 47, 47, 136, 47, 47, 
	47, 80, 47, 137, 47, 47, 47, 80, 
	47, 47, 138, 47, 47, 47, 80, 47, 
	47, 139, 47, 47, 47, 80, 47, 47, 
	140, 47, 47, 47, 80, 47, 141, 47, 
	47, 47, 80, 47, 47, 142, 47, 47, 
	47, 80, 47, 47, 143, 47, 47, 47, 
	80, 47, 47, 144, 47, 47, 47, 80, 
	47, 145, 47, 47, 47, 80, 47, 146, 
	47, 47, 47, 47, 80, 47, 147, 47, 
	47, 47, 47, 80, 47, 148, 47, 47, 
	47, 47, 80, 47, 47, 149, 47, 47, 
	47, 80, 47, 150, 47, 47, 47, 47, 
	80, 47, 151, 47, 47, 47, 47, 80, 
	47, 152, 47, 47, 47, 47, 80, 154, 
	153, 47, 47, 155, 47, 47, 47, 80, 
	47, 47, 156, 47, 47, 47, 80, 47, 
	47, 157, 47, 47, 47, 80, 47, 47, 
	158, 47, 47, 47, 80, 47, 47, 159, 
	47, 47, 47, 80, 47, 47, 160, 47, 
	47, 47, 80, 47, 47, 161, 47, 47, 
	47, 80, 47, 47, 162, 163, 47, 47, 
	47, 80, 47, 47, 164, 47, 47, 47, 
	80, 47, 47, 165, 47, 47, 47, 80, 
	47, 47, 166, 47, 47, 47, 80, 47, 
	47, 167, 47, 47, 47, 80, 47, 47, 
	168, 47, 47, 47, 80, 47, 47, 169, 
	47, 47, 47, 80, 47, 47, 170, 47, 
	47, 47, 80, 47, 47, 171, 172, 47, 
	47, 47, 80, 47, 47, 173, 47, 47, 
	47, 80, 47, 47, 174, 47, 47, 47, 
	80, 47, 47, 175, 47, 47, 47, 80, 
	47, 47, 176, 47, 47, 47, 80, 47, 
	47, 177, 47, 47, 47, 80, 47, 47, 
	178, 47, 47, 47, 80, 47, 47, 179, 
	47, 47, 47, 80, 47, 47, 180, 47, 
	47, 47, 80, 47, 47, 181, 47, 47, 
	47, 80, 47, 47, 182, 47, 47, 47, 
	80, 47, 47, 183, 47, 47, 47, 80, 
	47, 47, 184, 47, 47, 47, 80, 186, 
	185, 0
};

static const char _Scanner_trans_targs[] = {
	16, 1, 18, 2, 16, 3, 16, 4, 
	5, 0, 16, 6, 16, 7, 16, 9, 
	29, 32, 16, 12, 33, 16, 16, 14, 
	15, 16, 17, 19, 16, 21, 16, 16, 
	22, 16, 16, 23, 25, 16, 30, 31, 
	34, 16, 36, 37, 39, 16, 16, 27, 
	40, 44, 46, 77, 80, 16, 16, 81, 
	85, 88, 96, 97, 100, 105, 16, 109, 
	16, 16, 16, 16, 16, 20, 16, 16, 
	16, 16, 16, 16, 16, 24, 16, 16, 
	16, 26, 28, 26, 16, 8, 16, 16, 
	10, 16, 16, 11, 16, 35, 16, 16, 
	16, 16, 13, 16, 16, 38, 16, 16, 
	16, 16, 41, 42, 43, 27, 45, 27, 
	47, 73, 75, 76, 16, 48, 49, 58, 
	65, 69, 50, 51, 52, 53, 54, 55, 
	56, 57, 27, 59, 60, 61, 62, 63, 
	64, 27, 66, 67, 68, 27, 70, 71, 
	72, 27, 74, 27, 27, 27, 78, 79, 
	27, 16, 16, 82, 83, 84, 27, 86, 
	87, 27, 89, 90, 27, 91, 92, 93, 
	94, 95, 27, 27, 27, 98, 99, 27, 
	101, 102, 103, 104, 27, 106, 107, 108, 
	27, 16, 16
};

static const unsigned char _Scanner_trans_actions[] = {
	117, 0, 187, 0, 119, 0, 7, 0, 
	0, 0, 65, 0, 5, 0, 111, 0, 
	0, 3, 113, 0, 0, 115, 57, 0, 
	0, 9, 190, 0, 27, 0, 37, 39, 
	0, 17, 67, 0, 0, 21, 3, 3, 
	0, 69, 3, 0, 0, 59, 29, 184, 
	0, 0, 0, 0, 0, 43, 19, 0, 
	0, 0, 0, 0, 0, 0, 33, 0, 
	35, 25, 109, 107, 85, 0, 103, 63, 
	91, 49, 83, 23, 81, 0, 105, 55, 
	73, 184, 3, 181, 75, 0, 11, 13, 
	0, 77, 15, 0, 87, 0, 89, 31, 
	97, 53, 0, 45, 79, 0, 101, 61, 
	99, 47, 0, 0, 0, 133, 0, 136, 
	0, 0, 0, 0, 71, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 148, 0, 0, 0, 0, 0, 
	0, 142, 0, 0, 0, 151, 0, 0, 
	0, 145, 0, 127, 139, 124, 0, 0, 
	130, 95, 41, 0, 0, 0, 178, 0, 
	0, 172, 0, 0, 163, 0, 0, 0, 
	0, 0, 154, 166, 169, 0, 0, 175, 
	0, 0, 0, 0, 160, 0, 0, 0, 
	157, 93, 51
};

static const unsigned char _Scanner_to_state_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	121, 0, 0, 0, 0, 0, 0, 0, 
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
	0, 0, 0, 0, 0, 0
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
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0
};

static const short _Scanner_eof_trans[] = {
	0, 1, 5, 0, 0, 0, 0, 0, 
	15, 15, 15, 19, 19, 22, 0, 0, 
	0, 67, 68, 69, 71, 73, 75, 77, 
	79, 81, 5, 5, 85, 85, 85, 85, 
	90, 90, 93, 95, 97, 101, 103, 105, 
	81, 81, 81, 81, 81, 81, 81, 117, 
	81, 81, 81, 81, 81, 81, 81, 81, 
	81, 81, 81, 81, 81, 81, 81, 81, 
	81, 81, 81, 81, 81, 81, 81, 81, 
	81, 81, 81, 81, 81, 81, 81, 81, 
	154, 81, 81, 81, 81, 81, 81, 81, 
	81, 81, 81, 81, 81, 81, 81, 81, 
	81, 81, 81, 81, 81, 81, 81, 81, 
	81, 81, 81, 81, 81, 186
};

static const int Scanner_start = 16;
static const int Scanner_first_final = 16;
static const int Scanner_error = 0;

static const int Scanner_en_main = 16;


#line 140 "lexer.rl"


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
		if(tok != TOKEN_ELSE && 
            (nesting.size()==0 ||
                (nesting.top()!=TOKEN_LPAREN &&
                 nesting.top()!=TOKEN_LBRACKET &&
                 nesting.top()!=TOKEN_LBB)))
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

	if( tok == TOKEN_LPAREN ||
	    tok == TOKEN_LBRACKET ||
	    tok == TOKEN_LBB || 
	    tok == TOKEN_LBRACE)
        nesting.push(tok);
	else if(
        tok == TOKEN_RPAREN || 
        tok == TOKEN_RBRACE)
        nesting.pop();
    else if(
        tok == TOKEN_RBRACKET) {
        // need to do a bit of extra work to catch ]] vs. ]
        if(nesting.top() == TOKEN_LBRACKET) {
            nesting.pop();
        }
        else {
            nesting.pop();
            nesting.push(TOKEN_LBRACKET);
        }
    }
	else if(tok == TOKEN_FUNCTION)
        source.push(ts);

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
		std::cout << "Error (" << filename << ":" << intToStr(line+1) << "," 
            << intToStr(col+1) << ") : unexpected '" 
            << std::string(data, len) + "'" << std::endl; 
	}
}

Parser::Parser(Global& global, char const* filename) : line(0), col(0), global(global), filename(filename), errors(0), complete(false), lastTokenWasNL(false) 
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
	
#line 512 "lexer.cpp"
	{
	cs = Scanner_start;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 233 "lexer.rl"
	
#line 522 "lexer.cpp"
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
#line 543 "lexer.cpp"
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
				_trans += (unsigned int)(_mid - _keys);
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
				_trans += (unsigned int)((_mid - _keys)>>1);
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
#line 30 "lexer.rl"
	{act = 1;}
	break;
	case 5:
#line 31 "lexer.rl"
	{act = 2;}
	break;
	case 6:
#line 33 "lexer.rl"
	{act = 4;}
	break;
	case 7:
#line 34 "lexer.rl"
	{act = 5;}
	break;
	case 8:
#line 35 "lexer.rl"
	{act = 6;}
	break;
	case 9:
#line 36 "lexer.rl"
	{act = 7;}
	break;
	case 10:
#line 37 "lexer.rl"
	{act = 8;}
	break;
	case 11:
#line 38 "lexer.rl"
	{act = 9;}
	break;
	case 12:
#line 39 "lexer.rl"
	{act = 10;}
	break;
	case 13:
#line 41 "lexer.rl"
	{act = 11;}
	break;
	case 14:
#line 42 "lexer.rl"
	{act = 12;}
	break;
	case 15:
#line 43 "lexer.rl"
	{act = 13;}
	break;
	case 16:
#line 44 "lexer.rl"
	{act = 14;}
	break;
	case 17:
#line 45 "lexer.rl"
	{act = 15;}
	break;
	case 18:
#line 46 "lexer.rl"
	{act = 16;}
	break;
	case 19:
#line 47 "lexer.rl"
	{act = 17;}
	break;
	case 20:
#line 48 "lexer.rl"
	{act = 18;}
	break;
	case 21:
#line 49 "lexer.rl"
	{act = 19;}
	break;
	case 22:
#line 50 "lexer.rl"
	{act = 20;}
	break;
	case 23:
#line 60 "lexer.rl"
	{act = 23;}
	break;
	case 24:
#line 63 "lexer.rl"
	{act = 24;}
	break;
	case 25:
#line 134 "lexer.rl"
	{act = 72;}
	break;
	case 26:
#line 137 "lexer.rl"
	{act = 73;}
	break;
	case 27:
#line 54 "lexer.rl"
	{te = p+1;{std::string s(ts+1, te-ts-2); token( TOKEN_STR_CONST, Character::c(global.internStr(unescape(s))) );}}
	break;
	case 28:
#line 56 "lexer.rl"
	{te = p+1;{std::string s(ts+1, te-ts-2); token( TOKEN_STR_CONST, Character::c(global.internStr(unescape(s))) );}}
	break;
	case 29:
#line 65 "lexer.rl"
	{te = p+1;{std::string s(ts+1, te-ts-2); token( TOKEN_SYMBOL, CreateSymbol(global,global.internStr(unescape(s))) );}}
	break;
	case 30:
#line 71 "lexer.rl"
	{te = p+1;{token( TOKEN_NUM_CONST, Integer::c(strtoll(std::string(ts, te-1).c_str(), NULL, 10)) );}}
	break;
	case 31:
#line 74 "lexer.rl"
	{te = p+1;{token( TOKEN_NUM_CONST, CreateComplex(strtod(std::string(ts, te-1).c_str(), NULL)) );}}
	break;
	case 32:
#line 85 "lexer.rl"
	{te = p+1;{token( TOKEN_NUM_CONST, Integer::c(hexStrToInt(std::string(ts,te-ts))) );}}
	break;
	case 33:
#line 89 "lexer.rl"
	{te = p+1;{token( TOKEN_PLUS, CreateSymbol(global,Strings::add) );}}
	break;
	case 34:
#line 91 "lexer.rl"
	{te = p+1;{token( TOKEN_POW, CreateSymbol(global,Strings::pow) );}}
	break;
	case 35:
#line 92 "lexer.rl"
	{te = p+1;{token( TOKEN_DIVIDE, CreateSymbol(global,Strings::div) );}}
	break;
	case 36:
#line 94 "lexer.rl"
	{te = p+1;{token( TOKEN_POW, CreateSymbol(global,Strings::pow) );}}
	break;
	case 37:
#line 95 "lexer.rl"
	{te = p+1;{token( TOKEN_TILDE, CreateSymbol(global,Strings::tilde) );}}
	break;
	case 38:
#line 96 "lexer.rl"
	{te = p+1;{token( TOKEN_DOLLAR, CreateSymbol(global,Strings::dollar) );}}
	break;
	case 39:
#line 97 "lexer.rl"
	{te = p+1;{token( TOKEN_AT, CreateSymbol(global,Strings::at) );}}
	break;
	case 40:
#line 101 "lexer.rl"
	{te = p+1;{token( TOKEN_NS_GET_INT, CreateSymbol(global,Strings::nsgetint) );}}
	break;
	case 41:
#line 104 "lexer.rl"
	{te = p+1;{token( TOKEN_LBRACE, CreateSymbol(global,Strings::brace) );}}
	break;
	case 42:
#line 105 "lexer.rl"
	{te = p+1;{token( TOKEN_RBRACE );}}
	break;
	case 43:
#line 106 "lexer.rl"
	{te = p+1;{token( TOKEN_LPAREN, CreateSymbol(global,Strings::paren) );}}
	break;
	case 44:
#line 107 "lexer.rl"
	{te = p+1;{token( TOKEN_RPAREN );}}
	break;
	case 45:
#line 109 "lexer.rl"
	{te = p+1;{token( TOKEN_LBB, CreateSymbol(global,Strings::bb) );}}
	break;
	case 46:
#line 110 "lexer.rl"
	{te = p+1;{token( TOKEN_RBRACKET );}}
	break;
	case 47:
#line 113 "lexer.rl"
	{te = p+1;{token( TOKEN_LE, CreateSymbol(global,Strings::le) );}}
	break;
	case 48:
#line 114 "lexer.rl"
	{te = p+1;{token( TOKEN_GE, CreateSymbol(global,Strings::ge) );}}
	break;
	case 49:
#line 117 "lexer.rl"
	{te = p+1;{token( TOKEN_AND2, CreateSymbol(global,Strings::land2) );}}
	break;
	case 50:
#line 118 "lexer.rl"
	{te = p+1;{token( TOKEN_OR2, CreateSymbol(global,Strings::lor2) );}}
	break;
	case 51:
#line 119 "lexer.rl"
	{te = p+1;{token( TOKEN_LEFT_ASSIGN, CreateSymbol(global,Strings::assign) );}}
	break;
	case 52:
#line 121 "lexer.rl"
	{te = p+1;{token( TOKEN_RIGHT_ASSIGN, CreateSymbol(global,Strings::assign2) );}}
	break;
	case 53:
#line 122 "lexer.rl"
	{te = p+1;{token( TOKEN_LEFT_ASSIGN, CreateSymbol(global,Strings::assign2) );}}
	break;
	case 54:
#line 123 "lexer.rl"
	{te = p+1;{token( TOKEN_QUESTION, CreateSymbol(global,Strings::question) );}}
	break;
	case 55:
#line 126 "lexer.rl"
	{te = p+1;{token( TOKEN_SPECIALOP, CreateSymbol(global,Strings::id) );}}
	break;
	case 56:
#line 127 "lexer.rl"
	{te = p+1;{token( TOKEN_SPECIALOP, CreateSymbol(global,Strings::nid) );}}
	break;
	case 57:
#line 128 "lexer.rl"
	{te = p+1;{token(TOKEN_SPECIALOP, CreateSymbol(global,global.internStr(std::string(ts, te-ts))) ); }}
	break;
	case 58:
#line 131 "lexer.rl"
	{te = p+1;{token( TOKEN_COMMA );}}
	break;
	case 59:
#line 132 "lexer.rl"
	{te = p+1;{token( TOKEN_SEMICOLON );}}
	break;
	case 60:
#line 32 "lexer.rl"
	{te = p;p--;{token( TOKEN_NUM_CONST, Logical::NA() );}}
	break;
	case 61:
#line 63 "lexer.rl"
	{te = p;p--;{token( TOKEN_SYMBOL, CreateSymbol(global,global.internStr(std::string(ts, te-ts))) );}}
	break;
	case 62:
#line 68 "lexer.rl"
	{te = p;p--;{token( TOKEN_NUM_CONST, Double::c(strtod(std::string(ts, te).c_str(), NULL)) );}}
	break;
	case 63:
#line 82 "lexer.rl"
	{te = p;p--;{token( TOKEN_NUM_CONST, Double::c(hexStrToInt(std::string(ts,te-ts))) );}}
	break;
	case 64:
#line 88 "lexer.rl"
	{te = p;p--;{token( TOKEN_EQ_ASSIGN, CreateSymbol(global,Strings::eqassign) );}}
	break;
	case 65:
#line 90 "lexer.rl"
	{te = p;p--;{token( TOKEN_MINUS, CreateSymbol(global,Strings::sub) );}}
	break;
	case 66:
#line 93 "lexer.rl"
	{te = p;p--;{token( TOKEN_TIMES, CreateSymbol(global,Strings::mul) );}}
	break;
	case 67:
#line 98 "lexer.rl"
	{te = p;p--;{token( TOKEN_NOT, CreateSymbol(global,Strings::lnot) );}}
	break;
	case 68:
#line 99 "lexer.rl"
	{te = p;p--;{token( TOKEN_COLON, CreateSymbol(global,Strings::colon) );}}
	break;
	case 69:
#line 100 "lexer.rl"
	{te = p;p--;{token( TOKEN_NS_GET, CreateSymbol(global,Strings::nsget) );}}
	break;
	case 70:
#line 102 "lexer.rl"
	{te = p;p--;{token( TOKEN_AND, CreateSymbol(global,Strings::land) );}}
	break;
	case 71:
#line 103 "lexer.rl"
	{te = p;p--;{token( TOKEN_OR, CreateSymbol(global,Strings::lor) );}}
	break;
	case 72:
#line 108 "lexer.rl"
	{te = p;p--;{token( TOKEN_LBRACKET, CreateSymbol(global,Strings::bracket) );}}
	break;
	case 73:
#line 111 "lexer.rl"
	{te = p;p--;{token( TOKEN_LT, CreateSymbol(global,Strings::lt) );}}
	break;
	case 74:
#line 112 "lexer.rl"
	{te = p;p--;{token( TOKEN_GT, CreateSymbol(global,Strings::gt) );}}
	break;
	case 75:
#line 115 "lexer.rl"
	{te = p;p--;{token( TOKEN_EQ, CreateSymbol(global,Strings::eq) );}}
	break;
	case 76:
#line 116 "lexer.rl"
	{te = p;p--;{token( TOKEN_NE, CreateSymbol(global,Strings::neq) );}}
	break;
	case 77:
#line 120 "lexer.rl"
	{te = p;p--;{token( TOKEN_RIGHT_ASSIGN, CreateSymbol(global,Strings::assign) );}}
	break;
	case 78:
#line 134 "lexer.rl"
	{te = p;p--;{token( TOKEN_NEWLINE );}}
	break;
	case 79:
#line 137 "lexer.rl"
	{te = p;p--;}
	break;
	case 80:
#line 68 "lexer.rl"
	{{p = ((te))-1;}{token( TOKEN_NUM_CONST, Double::c(strtod(std::string(ts, te).c_str(), NULL)) );}}
	break;
	case 81:
#line 82 "lexer.rl"
	{{p = ((te))-1;}{token( TOKEN_NUM_CONST, Double::c(hexStrToInt(std::string(ts,te-ts))) );}}
	break;
	case 82:
#line 111 "lexer.rl"
	{{p = ((te))-1;}{token( TOKEN_LT, CreateSymbol(global,Strings::lt) );}}
	break;
	case 83:
#line 134 "lexer.rl"
	{{p = ((te))-1;}{token( TOKEN_NEWLINE );}}
	break;
	case 84:
#line 1 "NONE"
	{	switch( act ) {
	case 0:
	{{cs = 0; goto _again;}}
	break;
	case 1:
	{{p = ((te))-1;}token( TOKEN_NIL_CONST );}
	break;
	case 2:
	{{p = ((te))-1;}token( TOKEN_NULL_CONST, Null::Singleton() );}
	break;
	case 4:
	{{p = ((te))-1;}token( TOKEN_NUM_CONST, Logical::True() );}
	break;
	case 5:
	{{p = ((te))-1;}token( TOKEN_NUM_CONST, Logical::False() );}
	break;
	case 6:
	{{p = ((te))-1;}token( TOKEN_NUM_CONST, Double::Inf() );}
	break;
	case 7:
	{{p = ((te))-1;}token( TOKEN_NUM_CONST, Double::NaN() );}
	break;
	case 8:
	{{p = ((te))-1;}token( TOKEN_NUM_CONST, Integer::NA() );}
	break;
	case 9:
	{{p = ((te))-1;}token( TOKEN_NUM_CONST, Double::NA() );}
	break;
	case 10:
	{{p = ((te))-1;}token( TOKEN_STR_CONST, Character::NA() );}
	break;
	case 11:
	{{p = ((te))-1;}token( TOKEN_STR_CONST, List::NA() );}
	break;
	case 12:
	{{p = ((te))-1;}token( TOKEN_FUNCTION, CreateSymbol(global,Strings::function) );}
	break;
	case 13:
	{{p = ((te))-1;}token( TOKEN_WHILE, CreateSymbol(global,Strings::whileSym) );}
	break;
	case 14:
	{{p = ((te))-1;}token( TOKEN_REPEAT, CreateSymbol(global,Strings::repeatSym) );}
	break;
	case 15:
	{{p = ((te))-1;}token( TOKEN_FOR, CreateSymbol(global,Strings::forSym) );}
	break;
	case 16:
	{{p = ((te))-1;}token( TOKEN_IF, CreateSymbol(global,Strings::ifSym) );}
	break;
	case 17:
	{{p = ((te))-1;}token( TOKEN_IN );}
	break;
	case 18:
	{{p = ((te))-1;}token( TOKEN_ELSE );}
	break;
	case 19:
	{{p = ((te))-1;}token( TOKEN_NEXT, CreateSymbol(global,Strings::nextSym) );}
	break;
	case 20:
	{{p = ((te))-1;}token( TOKEN_BREAK, CreateSymbol(global,Strings::breakSym) );}
	break;
	case 23:
	{{p = ((te))-1;}token( TOKEN_SYMBOL, CreateSymbol(global,global.internStr(std::string(ts, te-ts))));}
	break;
	case 24:
	{{p = ((te))-1;}token( TOKEN_SYMBOL, CreateSymbol(global,global.internStr(std::string(ts, te-ts))) );}
	break;
	case 72:
	{{p = ((te))-1;}token( TOKEN_NEWLINE );}
	break;
	default:
	{{p = ((te))-1;}}
	break;
	}
	}
	break;
#line 1011 "lexer.cpp"
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
#line 1028 "lexer.cpp"
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

#line 234 "lexer.rl"
	int syntaxErrors = errors;
	Parse(pParser, 0, Value::Nil(), this);
	ParseFree(pParser, free);
	errors = syntaxErrors;

	if( cs == Scanner_error && syntaxErrors == 0) {
		syntaxErrors++;
		std::cout << "Lexing error (" << filename << ":" << intToStr(line+1) << ":" << intToStr(col+1) << ")" << std::endl;
        // TODO: make a meaningful error. te can be 0x0 sometimes! Try entering `z
        // (" << intToStr(line+1) << "," << intToStr(col+1) << ") : unexpected '" << std::string(ts, te-ts) + "'" << std::endl; 
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

String Parser::popSource() {
	assert(source.size() > 0);
	std::string s(source.top(), le-source.top());
	String result = global.internStr(rtrim(s));
	source.pop();
	return result;	
}

int parse(Global& global, char const* filename,
    char const* code, size_t len, bool isEof, Value& result, FILE* trace) {
    Parser parser(global, filename);
    return parser.execute(code, len, isEof, result, trace);
}


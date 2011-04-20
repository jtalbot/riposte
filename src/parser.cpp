
#line 1 "lexer.rl"
/*
 *	A ragel lexer for R.
 *	In ragel, the lexer drives the parsing process, so this also has the basic parsing functions.
 *	Use this to generate parser.cpp
 *      TODO: Eliminate calls to Symbol(state, ...) which require a map search. Should be a hard-coded value.
 *            Parse escape sequences embedded in strings.
 *            Emit complex NA
 *            Include the double-to-int warnings, e.g. on 1.0L  and 1.5L
 *            Generate hex numbers
 *            Do we really want to allow '.' inside hex numbers? R allows them, but ignores them when parsing to a number.
 *            R parser has a rule for OP % OP. Is this ever used?
 */

#include "parser.h"
#include "tokens.h"
#include "ast.cpp"


#line 22 "../parser.cpp"
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
	80, 1, 81, 1, 82, 2, 0, 1, 
	2, 3, 4, 2, 3, 5, 2, 3, 
	6, 2, 3, 7, 2, 3, 8, 2, 
	3, 9, 2, 3, 10, 2, 3, 11, 
	2, 3, 12, 2, 3, 13, 2, 3, 
	14, 2, 3, 15, 2, 3, 16, 2, 
	3, 17, 2, 3, 18, 2, 3, 19, 
	2, 3, 20, 2, 3, 21, 2, 3, 
	22, 2, 3, 23, 2, 3, 24, 2, 
	3, 25, 2, 3, 26
};

static const short _Scanner_key_offsets[] = {
	0, 0, 4, 5, 8, 8, 10, 13, 
	13, 15, 19, 25, 29, 31, 32, 35, 
	35, 83, 87, 91, 92, 94, 95, 96, 
	97, 98, 106, 114, 124, 134, 138, 146, 
	154, 160, 167, 175, 177, 178, 179, 182, 
	183, 184, 193, 202, 211, 220, 229, 238, 
	249, 257, 268, 278, 287, 296, 305, 314, 
	323, 332, 341, 349, 358, 367, 376, 385, 
	394, 402, 411, 420, 429, 438, 447, 456, 
	464, 473, 482, 491, 499, 508, 517, 526, 
	535, 544, 553, 554, 555, 564, 573, 582, 
	591, 600, 609, 618, 628, 637, 646, 655, 
	664, 673, 682, 691, 701, 710, 719, 728, 
	737, 746, 755, 764, 773, 782, 791, 800, 
	809
};

static const char _Scanner_trans_keys[] = {
	10, 35, 33, 126, 10, 10, 34, 92, 
	10, 37, 10, 39, 92, 48, 57, 43, 
	45, 48, 57, 48, 57, 65, 70, 97, 
	102, 43, 45, 48, 57, 48, 57, 45, 
	10, 92, 96, 10, 33, 34, 35, 36, 
	37, 38, 39, 40, 41, 42, 43, 44, 
	45, 46, 47, 48, 58, 59, 60, 61, 
	62, 63, 64, 70, 73, 78, 84, 91, 
	92, 93, 94, 96, 98, 101, 102, 105, 
	110, 114, 119, 123, 124, 125, 126, 49, 
	57, 65, 122, 10, 35, 33, 126, 10, 
	35, 33, 126, 61, 10, 37, 38, 42, 
	62, 62, 46, 95, 48, 57, 65, 90, 
	97, 122, 46, 95, 48, 57, 65, 90, 
	97, 122, 46, 69, 95, 101, 48, 57, 
	65, 90, 97, 122, 43, 45, 46, 95, 
	48, 57, 65, 90, 97, 122, 76, 105, 
	48, 57, 46, 95, 48, 57, 65, 90, 
	97, 122, 46, 69, 76, 101, 105, 120, 
	48, 57, 69, 76, 101, 105, 48, 57, 
	46, 69, 76, 101, 105, 48, 57, 80, 
	112, 48, 57, 65, 70, 97, 102, 48, 
	57, 58, 58, 45, 60, 61, 61, 61, 
	46, 65, 95, 48, 57, 66, 90, 97, 
	122, 46, 76, 95, 48, 57, 65, 90, 
	97, 122, 46, 83, 95, 48, 57, 65, 
	90, 97, 122, 46, 69, 95, 48, 57, 
	65, 90, 97, 122, 46, 95, 110, 48, 
	57, 65, 90, 97, 122, 46, 95, 102, 
	48, 57, 65, 90, 97, 122, 46, 65, 
	85, 95, 97, 48, 57, 66, 90, 98, 
	122, 46, 95, 48, 57, 65, 90, 97, 
	122, 46, 95, 99, 105, 114, 48, 57, 
	65, 90, 97, 122, 46, 95, 104, 111, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	97, 48, 57, 65, 90, 98, 122, 46, 
	95, 114, 48, 57, 65, 90, 97, 122, 
	46, 95, 97, 48, 57, 65, 90, 98, 
	122, 46, 95, 99, 48, 57, 65, 90, 
	97, 122, 46, 95, 116, 48, 57, 65, 
	90, 97, 122, 46, 95, 101, 48, 57, 
	65, 90, 97, 122, 46, 95, 114, 48, 
	57, 65, 90, 97, 122, 46, 95, 48, 
	57, 65, 90, 97, 122, 46, 95, 109, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	112, 48, 57, 65, 90, 97, 122, 46, 
	95, 108, 48, 57, 65, 90, 97, 122, 
	46, 95, 101, 48, 57, 65, 90, 97, 
	122, 46, 95, 120, 48, 57, 65, 90, 
	97, 122, 46, 95, 48, 57, 65, 90, 
	97, 122, 46, 95, 110, 48, 57, 65, 
	90, 97, 122, 46, 95, 116, 48, 57, 
	65, 90, 97, 122, 46, 95, 101, 48, 
	57, 65, 90, 97, 122, 46, 95, 103, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	101, 48, 57, 65, 90, 97, 122, 46, 
	95, 114, 48, 57, 65, 90, 97, 122, 
	46, 95, 48, 57, 65, 90, 97, 122, 
	46, 95, 101, 48, 57, 65, 90, 97, 
	122, 46, 95, 97, 48, 57, 65, 90, 
	98, 122, 46, 95, 108, 48, 57, 65, 
	90, 97, 122, 46, 95, 48, 57, 65, 
	90, 97, 122, 46, 76, 95, 48, 57, 
	65, 90, 97, 122, 46, 76, 95, 48, 
	57, 65, 90, 97, 122, 46, 78, 95, 
	48, 57, 65, 90, 97, 122, 46, 82, 
	95, 48, 57, 65, 90, 97, 122, 46, 
	85, 95, 48, 57, 65, 90, 97, 122, 
	46, 69, 95, 48, 57, 65, 90, 97, 
	122, 91, 93, 46, 95, 114, 48, 57, 
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
	0, 2, 0, 2, 0, 1, 3, 0, 
	44, 2, 2, 1, 2, 1, 1, 1, 
	1, 2, 2, 4, 4, 2, 2, 6, 
	4, 5, 2, 0, 1, 1, 3, 1, 
	1, 3, 3, 3, 3, 3, 3, 5, 
	2, 5, 4, 3, 3, 3, 3, 3, 
	3, 3, 2, 3, 3, 3, 3, 3, 
	2, 3, 3, 3, 3, 3, 3, 2, 
	3, 3, 3, 2, 3, 3, 3, 3, 
	3, 3, 1, 1, 3, 3, 3, 3, 
	3, 3, 3, 4, 3, 3, 3, 3, 
	3, 3, 3, 4, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	1
};

static const char _Scanner_range_lengths[] = {
	0, 1, 0, 0, 0, 0, 0, 0, 
	1, 1, 3, 1, 1, 0, 0, 0, 
	2, 1, 1, 0, 0, 0, 0, 0, 
	0, 3, 3, 3, 3, 1, 3, 1, 
	1, 1, 3, 1, 0, 0, 0, 0, 
	0, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 0, 0, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	0
};

static const short _Scanner_index_offsets[] = {
	0, 0, 4, 6, 10, 11, 14, 18, 
	19, 21, 25, 29, 33, 35, 37, 41, 
	42, 89, 93, 97, 99, 102, 104, 106, 
	108, 110, 116, 122, 130, 138, 142, 148, 
	156, 162, 169, 175, 177, 179, 181, 185, 
	187, 189, 196, 203, 210, 217, 224, 231, 
	240, 246, 255, 263, 270, 277, 284, 291, 
	298, 305, 312, 318, 325, 332, 339, 346, 
	353, 359, 366, 373, 380, 387, 394, 401, 
	407, 414, 421, 428, 434, 441, 448, 455, 
	462, 469, 476, 478, 480, 487, 494, 501, 
	508, 515, 522, 529, 537, 544, 551, 558, 
	565, 572, 579, 586, 594, 601, 608, 615, 
	622, 629, 636, 643, 650, 657, 664, 671, 
	678
};

static const unsigned char _Scanner_indicies[] = {
	2, 3, 0, 1, 2, 3, 6, 7, 
	8, 5, 5, 4, 10, 9, 6, 12, 
	13, 11, 11, 14, 4, 16, 16, 14, 
	15, 17, 17, 17, 15, 19, 19, 20, 
	18, 20, 18, 22, 21, 6, 24, 25, 
	23, 23, 2, 27, 5, 3, 28, 9, 
	29, 11, 30, 31, 32, 33, 34, 35, 
	36, 37, 38, 40, 41, 42, 43, 44, 
	45, 46, 48, 49, 50, 51, 52, 6, 
	53, 54, 23, 55, 56, 57, 58, 59, 
	60, 61, 62, 63, 64, 65, 39, 47, 
	26, 2, 3, 66, 26, 2, 3, 67, 
	1, 69, 68, 70, 10, 9, 72, 71, 
	74, 73, 76, 75, 78, 77, 47, 47, 
	80, 47, 47, 79, 47, 47, 47, 47, 
	47, 4, 47, 81, 47, 81, 80, 47, 
	47, 79, 16, 16, 47, 47, 82, 47, 
	47, 79, 84, 85, 14, 83, 47, 47, 
	82, 47, 47, 79, 86, 87, 84, 87, 
	85, 88, 39, 83, 87, 84, 87, 85, 
	86, 83, 86, 87, 84, 87, 85, 39, 
	83, 90, 90, 17, 17, 17, 89, 20, 
	89, 92, 91, 94, 93, 96, 97, 98, 
	95, 100, 99, 102, 101, 47, 103, 47, 
	47, 47, 47, 79, 47, 104, 47, 47, 
	47, 47, 79, 47, 105, 47, 47, 47, 
	47, 79, 47, 106, 47, 47, 47, 47, 
	79, 47, 47, 107, 47, 47, 47, 79, 
	47, 47, 108, 47, 47, 47, 79, 47, 
	109, 110, 47, 111, 47, 47, 47, 79, 
	47, 113, 47, 47, 47, 112, 47, 47, 
	114, 115, 116, 47, 47, 47, 79, 47, 
	47, 117, 118, 47, 47, 47, 79, 47, 
	47, 119, 47, 47, 47, 79, 47, 47, 
	120, 47, 47, 47, 79, 47, 47, 121, 
	47, 47, 47, 79, 47, 47, 122, 47, 
	47, 47, 79, 47, 47, 123, 47, 47, 
	47, 79, 47, 47, 124, 47, 47, 47, 
	79, 47, 47, 125, 47, 47, 47, 79, 
	47, 126, 47, 47, 47, 79, 47, 47, 
	127, 47, 47, 47, 79, 47, 47, 128, 
	47, 47, 47, 79, 47, 47, 129, 47, 
	47, 47, 79, 47, 47, 130, 47, 47, 
	47, 79, 47, 47, 131, 47, 47, 47, 
	79, 47, 132, 47, 47, 47, 79, 47, 
	47, 133, 47, 47, 47, 79, 47, 47, 
	134, 47, 47, 47, 79, 47, 47, 135, 
	47, 47, 47, 79, 47, 47, 136, 47, 
	47, 47, 79, 47, 47, 137, 47, 47, 
	47, 79, 47, 47, 138, 47, 47, 47, 
	79, 47, 139, 47, 47, 47, 79, 47, 
	47, 140, 47, 47, 47, 79, 47, 47, 
	141, 47, 47, 47, 79, 47, 47, 142, 
	47, 47, 47, 79, 47, 143, 47, 47, 
	47, 79, 47, 144, 47, 47, 47, 47, 
	79, 47, 145, 47, 47, 47, 47, 79, 
	47, 146, 47, 47, 47, 47, 79, 47, 
	147, 47, 47, 47, 47, 79, 47, 148, 
	47, 47, 47, 47, 79, 47, 149, 47, 
	47, 47, 47, 79, 151, 150, 153, 152, 
	47, 47, 154, 47, 47, 47, 79, 47, 
	47, 155, 47, 47, 47, 79, 47, 47, 
	156, 47, 47, 47, 79, 47, 47, 157, 
	47, 47, 47, 79, 47, 47, 158, 47, 
	47, 47, 79, 47, 47, 159, 47, 47, 
	47, 79, 47, 47, 160, 47, 47, 47, 
	79, 47, 47, 161, 162, 47, 47, 47, 
	79, 47, 47, 163, 47, 47, 47, 79, 
	47, 47, 164, 47, 47, 47, 79, 47, 
	47, 165, 47, 47, 47, 79, 47, 47, 
	166, 47, 47, 47, 79, 47, 47, 167, 
	47, 47, 47, 79, 47, 47, 168, 47, 
	47, 47, 79, 47, 47, 169, 47, 47, 
	47, 79, 47, 47, 170, 171, 47, 47, 
	47, 79, 47, 47, 172, 47, 47, 47, 
	79, 47, 47, 173, 47, 47, 47, 79, 
	47, 47, 174, 47, 47, 47, 79, 47, 
	47, 175, 47, 47, 47, 79, 47, 47, 
	176, 47, 47, 47, 79, 47, 47, 177, 
	47, 47, 47, 79, 47, 47, 178, 47, 
	47, 47, 79, 47, 47, 179, 47, 47, 
	47, 79, 47, 47, 180, 47, 47, 47, 
	79, 47, 47, 181, 47, 47, 47, 79, 
	47, 47, 182, 47, 47, 47, 79, 47, 
	47, 183, 47, 47, 47, 79, 185, 184, 
	0
};

static const char _Scanner_trans_targs[] = {
	16, 1, 18, 2, 16, 3, 0, 16, 
	4, 5, 20, 6, 16, 7, 29, 16, 
	8, 34, 16, 12, 35, 16, 16, 14, 
	15, 16, 17, 19, 16, 21, 16, 16, 
	22, 16, 16, 23, 25, 16, 31, 33, 
	36, 16, 38, 39, 40, 16, 16, 26, 
	41, 45, 47, 79, 82, 83, 16, 84, 
	88, 91, 99, 100, 103, 108, 16, 112, 
	16, 16, 16, 16, 16, 16, 16, 16, 
	16, 16, 16, 16, 24, 16, 16, 16, 
	27, 28, 30, 16, 16, 16, 32, 9, 
	10, 16, 11, 16, 37, 16, 16, 16, 
	16, 13, 16, 16, 16, 16, 16, 42, 
	43, 44, 26, 46, 26, 48, 76, 78, 
	16, 49, 50, 65, 72, 51, 59, 52, 
	53, 54, 55, 56, 57, 58, 26, 60, 
	61, 62, 63, 64, 26, 66, 67, 68, 
	69, 70, 71, 26, 73, 74, 75, 26, 
	77, 26, 26, 80, 81, 26, 16, 16, 
	16, 16, 85, 86, 87, 26, 89, 90, 
	26, 92, 93, 26, 94, 95, 96, 97, 
	98, 26, 26, 26, 101, 102, 26, 104, 
	105, 106, 107, 26, 109, 110, 111, 26, 
	16, 16
};

static const unsigned char _Scanner_trans_actions[] = {
	113, 0, 183, 0, 115, 0, 0, 7, 
	0, 0, 180, 0, 5, 0, 0, 107, 
	0, 3, 109, 0, 0, 111, 59, 0, 
	0, 9, 186, 0, 25, 0, 35, 37, 
	0, 15, 63, 0, 0, 19, 177, 177, 
	0, 65, 3, 0, 0, 61, 27, 174, 
	0, 0, 0, 0, 0, 0, 17, 0, 
	0, 0, 0, 0, 0, 0, 31, 0, 
	33, 23, 105, 103, 81, 49, 101, 87, 
	51, 79, 21, 77, 0, 99, 57, 69, 
	0, 174, 0, 71, 13, 11, 177, 0, 
	0, 73, 0, 83, 0, 85, 29, 95, 
	55, 0, 43, 75, 47, 97, 45, 0, 
	0, 0, 126, 0, 129, 0, 0, 0, 
	67, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 141, 0, 
	0, 0, 0, 0, 144, 0, 0, 0, 
	0, 0, 0, 135, 0, 0, 0, 138, 
	0, 120, 132, 0, 0, 123, 91, 39, 
	93, 41, 0, 0, 0, 171, 0, 0, 
	165, 0, 0, 156, 0, 0, 0, 0, 
	0, 147, 159, 162, 0, 0, 168, 0, 
	0, 0, 0, 153, 0, 0, 0, 150, 
	89, 53
};

static const unsigned char _Scanner_to_state_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	117, 0, 0, 0, 0, 0, 0, 0, 
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
	0, 0, 0, 0, 0, 0, 0, 0, 
	0
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
	0, 0, 0, 0, 0, 0, 0, 0, 
	0
};

static const short _Scanner_eof_trans[] = {
	0, 1, 5, 0, 0, 5, 0, 0, 
	5, 16, 16, 19, 19, 22, 0, 0, 
	0, 67, 68, 69, 71, 72, 74, 76, 
	78, 80, 5, 80, 80, 84, 80, 84, 
	84, 84, 90, 90, 92, 94, 96, 100, 
	102, 80, 80, 80, 80, 80, 80, 80, 
	113, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 151, 153, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	185
};

static const int Scanner_start = 16;
static const int Scanner_first_final = 16;
static const int Scanner_error = 0;

static const int Scanner_en_main = 16;


#line 132 "lexer.rl"


void Parser::token( int tok, Value v)
{
	Parser::Result result;
	const char *data = ts;
	int len = te - ts;

	/*std::cout << '<' << tok << "> ";
	std::cout.write( data, len );
	std::cout << '\n';*/

	// Do the lookahead to resolve the dangling else conflict
	if(lastTokenWasNL) {
		if(tok != TOKEN_ELSE)
			Parse(pParser, TOKEN_NEWLINE, Value::NIL, &result);
		Parse(pParser, tok, v, &result);
		lastTokenWasNL = false;
	}
	else {
		if(tok == TOKEN_NEWLINE)
			lastTokenWasNL = true;
		else
			Parse(pParser, tok, v, &result);
	}

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
}

Parser::Parser(State& state) : line(0), col(0), have(0), state(state), lastTokenWasNL(false)
{
	
#line 473 "../parser.cpp"
	{
	cs = Scanner_start;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 173 "lexer.rl"
}

int Parser::execute( const char* data, int len, bool isEof, Value& result)
{
	Result r;
	r.state = 0;

	pParser = ParseAlloc(malloc);

	const char *p = data;
	const char *pe = data+len;
	const char* eof = isEof ? pe : 0;

	lastTokenWasNL = false;

	
#line 498 "../parser.cpp"
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
#line 519 "../parser.cpp"
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
#line 30 "lexer.rl"
	{act = 1;}
	break;
	case 5:
#line 32 "lexer.rl"
	{act = 3;}
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
#line 48 "lexer.rl"
	{act = 19;}
	break;
	case 22:
#line 58 "lexer.rl"
	{act = 22;}
	break;
	case 23:
#line 64 "lexer.rl"
	{act = 24;}
	break;
	case 24:
#line 120 "lexer.rl"
	{act = 65;}
	break;
	case 25:
#line 126 "lexer.rl"
	{act = 68;}
	break;
	case 26:
#line 129 "lexer.rl"
	{act = 69;}
	break;
	case 27:
#line 52 "lexer.rl"
	{te = p+1;{token( TOKEN_STR_CONST, Character::c(state, std::string(ts+1, te-ts-2)) );}}
	break;
	case 28:
#line 54 "lexer.rl"
	{te = p+1;{token( TOKEN_STR_CONST, Character::c(state, std::string(ts+1, te-ts-2)) );}}
	break;
	case 29:
#line 60 "lexer.rl"
	{te = p+1;{token( TOKEN_SYMBOL, Symbol(state, std::string(ts+1, te-ts-2)) );}}
	break;
	case 30:
#line 67 "lexer.rl"
	{te = p+1;{token( TOKEN_NUM_CONST, Complex::c(0, atof(std::string(ts, te-ts-1).c_str())) );}}
	break;
	case 31:
#line 70 "lexer.rl"
	{te = p+1;{token( TOKEN_NUM_CONST, Integer::c(atof(std::string(ts, te-ts-1).c_str())) );}}
	break;
	case 32:
#line 82 "lexer.rl"
	{te = p+1;{token( TOKEN_PLUS, Symbol(state, "+") );}}
	break;
	case 33:
#line 84 "lexer.rl"
	{te = p+1;{token( TOKEN_POW, Symbol(state, "^") );}}
	break;
	case 34:
#line 85 "lexer.rl"
	{te = p+1;{token( TOKEN_DIVIDE, Symbol(state, "/") );}}
	break;
	case 35:
#line 87 "lexer.rl"
	{te = p+1;{token( TOKEN_POW, Symbol(state, "^") );}}
	break;
	case 36:
#line 88 "lexer.rl"
	{te = p+1;{token( TOKEN_TILDE, Symbol(state, "~") );}}
	break;
	case 37:
#line 89 "lexer.rl"
	{te = p+1;{token( TOKEN_DOLLAR, Symbol(state, "$") );}}
	break;
	case 38:
#line 90 "lexer.rl"
	{te = p+1;{token( TOKEN_AT, Symbol(state, "@") );}}
	break;
	case 39:
#line 94 "lexer.rl"
	{te = p+1;{token( TOKEN_NS_GET_INT, Symbol(state, ":::") );}}
	break;
	case 40:
#line 97 "lexer.rl"
	{te = p+1;{token( TOKEN_LBRACE, Symbol(state, "{") );}}
	break;
	case 41:
#line 98 "lexer.rl"
	{te = p+1;{token( TOKEN_RBRACE, Symbol(state, "}") );}}
	break;
	case 42:
#line 99 "lexer.rl"
	{te = p+1;{token( TOKEN_LPAREN, Symbol(state, "(") );}}
	break;
	case 43:
#line 100 "lexer.rl"
	{te = p+1;{token( TOKEN_RPAREN, Symbol(state, ")") );}}
	break;
	case 44:
#line 102 "lexer.rl"
	{te = p+1;{token( TOKEN_LBB, Symbol(state, "[[") );}}
	break;
	case 45:
#line 104 "lexer.rl"
	{te = p+1;{token( TOKEN_RBB, Symbol(state, "]]") );}}
	break;
	case 46:
#line 107 "lexer.rl"
	{te = p+1;{token( TOKEN_LE, Symbol(state, "<=") );}}
	break;
	case 47:
#line 108 "lexer.rl"
	{te = p+1;{token( TOKEN_GE, Symbol(state, ">=") );}}
	break;
	case 48:
#line 109 "lexer.rl"
	{te = p+1;{token( TOKEN_EQ, Symbol(state, "==") );}}
	break;
	case 49:
#line 110 "lexer.rl"
	{te = p+1;{token( TOKEN_NE, Symbol(state, "!=") );}}
	break;
	case 50:
#line 111 "lexer.rl"
	{te = p+1;{token( TOKEN_AND2, Symbol(state, "&&") );}}
	break;
	case 51:
#line 112 "lexer.rl"
	{te = p+1;{token( TOKEN_OR2, Symbol(state, "||") );}}
	break;
	case 52:
#line 113 "lexer.rl"
	{te = p+1;{token( TOKEN_LEFT_ASSIGN, Symbol(state, "<-") );}}
	break;
	case 53:
#line 115 "lexer.rl"
	{te = p+1;{token( TOKEN_RIGHT_ASSIGN, Symbol(state, "->>") );}}
	break;
	case 54:
#line 116 "lexer.rl"
	{te = p+1;{token( TOKEN_LEFT_ASSIGN, Symbol(state, "<<-") );}}
	break;
	case 55:
#line 117 "lexer.rl"
	{te = p+1;{token( TOKEN_QUESTION, Symbol(state, "?") );}}
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
#line 31 "lexer.rl"
	{te = p;p--;{token( TOKEN_NUM_CONST, Logical::NA );}}
	break;
	case 59:
#line 58 "lexer.rl"
	{te = p;p--;{token( TOKEN_SYMBOL, Symbol(state, std::string(ts, te-ts)) );}}
	break;
	case 60:
#line 64 "lexer.rl"
	{te = p;p--;{token( TOKEN_NUM_CONST, Double::c(atof(std::string(ts, te-ts).c_str())) );}}
	break;
	case 61:
#line 78 "lexer.rl"
	{te = p;p--;{token( TOKEN_NUM_CONST );}}
	break;
	case 62:
#line 81 "lexer.rl"
	{te = p;p--;{token( TOKEN_EQ_ASSIGN, Symbol(state, "=") );}}
	break;
	case 63:
#line 83 "lexer.rl"
	{te = p;p--;{token( TOKEN_MINUS, Symbol(state, "-") );}}
	break;
	case 64:
#line 86 "lexer.rl"
	{te = p;p--;{token( TOKEN_TIMES, Symbol(state, "*") );}}
	break;
	case 65:
#line 91 "lexer.rl"
	{te = p;p--;{token( TOKEN_NOT, Symbol(state, "!") );}}
	break;
	case 66:
#line 92 "lexer.rl"
	{te = p;p--;{token( TOKEN_COLON, Symbol(state, ":") );}}
	break;
	case 67:
#line 93 "lexer.rl"
	{te = p;p--;{token( TOKEN_NS_GET, Symbol(state, "::") );}}
	break;
	case 68:
#line 95 "lexer.rl"
	{te = p;p--;{token( TOKEN_AND, Symbol(state, "&") );}}
	break;
	case 69:
#line 96 "lexer.rl"
	{te = p;p--;{token( TOKEN_OR, Symbol(state, "|") );}}
	break;
	case 70:
#line 101 "lexer.rl"
	{te = p;p--;{token( TOKEN_LBRACKET, Symbol(state, "[") );}}
	break;
	case 71:
#line 103 "lexer.rl"
	{te = p;p--;{token( TOKEN_RBRACKET, Symbol(state, "]") );}}
	break;
	case 72:
#line 105 "lexer.rl"
	{te = p;p--;{token( TOKEN_LT, Symbol(state, "<") );}}
	break;
	case 73:
#line 106 "lexer.rl"
	{te = p;p--;{token( TOKEN_GT, Symbol(state, ">") );}}
	break;
	case 74:
#line 114 "lexer.rl"
	{te = p;p--;{token( TOKEN_RIGHT_ASSIGN, Symbol(state, "->") );}}
	break;
	case 75:
#line 120 "lexer.rl"
	{te = p;p--;{token(TOKEN_SPECIALOP, Symbol(state, std::string(ts, te-ts)) ); }}
	break;
	case 76:
#line 126 "lexer.rl"
	{te = p;p--;{token( TOKEN_NEWLINE );}}
	break;
	case 77:
#line 129 "lexer.rl"
	{te = p;p--;}
	break;
	case 78:
#line 64 "lexer.rl"
	{{p = ((te))-1;}{token( TOKEN_NUM_CONST, Double::c(atof(std::string(ts, te-ts).c_str())) );}}
	break;
	case 79:
#line 78 "lexer.rl"
	{{p = ((te))-1;}{token( TOKEN_NUM_CONST );}}
	break;
	case 80:
#line 105 "lexer.rl"
	{{p = ((te))-1;}{token( TOKEN_LT, Symbol(state, "<") );}}
	break;
	case 81:
#line 126 "lexer.rl"
	{{p = ((te))-1;}{token( TOKEN_NEWLINE );}}
	break;
	case 82:
#line 1 "NONE"
	{	switch( act ) {
	case 0:
	{{cs = 0; goto _again;}}
	break;
	case 1:
	{{p = ((te))-1;}token( TOKEN_NULL_CONST, Null::singleton );}
	break;
	case 3:
	{{p = ((te))-1;}token( TOKEN_NUM_CONST, Logical::True );}
	break;
	case 4:
	{{p = ((te))-1;}token( TOKEN_NUM_CONST, Logical::False );}
	break;
	case 5:
	{{p = ((te))-1;}token( TOKEN_NUM_CONST, Double::Inf);}
	break;
	case 6:
	{{p = ((te))-1;}token( TOKEN_NUM_CONST, Double::NaN);}
	break;
	case 7:
	{{p = ((te))-1;}token( TOKEN_NUM_CONST, Integer::NA);}
	break;
	case 8:
	{{p = ((te))-1;}token( TOKEN_NUM_CONST, Double::NA);}
	break;
	case 9:
	{{p = ((te))-1;}token( TOKEN_STR_CONST, Character::NA);}
	break;
	case 10:
	{{p = ((te))-1;}token( TOKEN_NUM_CONST);}
	break;
	case 11:
	{{p = ((te))-1;}token( TOKEN_FUNCTION, Symbol(state, "function") );}
	break;
	case 12:
	{{p = ((te))-1;}token( TOKEN_WHILE, Symbol(state, "while") );}
	break;
	case 13:
	{{p = ((te))-1;}token( TOKEN_REPEAT, Symbol(state, "repeat") );}
	break;
	case 14:
	{{p = ((te))-1;}token( TOKEN_FOR, Symbol(state, "for") );}
	break;
	case 15:
	{{p = ((te))-1;}token( TOKEN_IF, Symbol(state, "if") );}
	break;
	case 16:
	{{p = ((te))-1;}token( TOKEN_IN, Symbol(state, "in") );}
	break;
	case 17:
	{{p = ((te))-1;}token( TOKEN_ELSE, Symbol(state, "else") );}
	break;
	case 18:
	{{p = ((te))-1;}token( TOKEN_NEXT, Symbol(state, "next") );}
	break;
	case 19:
	{{p = ((te))-1;}token( TOKEN_BREAK, Symbol(state, "break") );}
	break;
	case 22:
	{{p = ((te))-1;}token( TOKEN_SYMBOL, Symbol(state, std::string(ts, te-ts)) );}
	break;
	case 24:
	{{p = ((te))-1;}token( TOKEN_NUM_CONST, Double::c(atof(std::string(ts, te-ts).c_str())) );}
	break;
	case 65:
	{{p = ((te))-1;}token(TOKEN_SPECIALOP, Symbol(state, std::string(ts, te-ts)) ); }
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
#line 979 "../parser.cpp"
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
#line 996 "../parser.cpp"
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

#line 189 "lexer.rl"

	Parse(pParser, 0, Value::NIL, &r);
	ParseFree(pParser, free);

	result = r.value;

	if( cs == Scanner_error || r.state == -1 )
		return -1;
	else if( cs >= Scanner_first_final && r.state == 1)
		return 1;
	else
		return 0;
}

int Parser::buffer_execute( )
{
	/*static char buf[16384];

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
	*/
	return 0;
}

int Parser::finish()
{
	if( cs == Scanner_error )
		return -1;
	else if( cs >= Scanner_first_final )
		return 1;
	else
		return 0;
}


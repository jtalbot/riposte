
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
	0, 1, 0, 1, 1, 1, 3, 1, 
	4, 1, 29, 1, 30, 1, 31, 1, 
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
	80, 1, 81, 1, 82, 1, 83, 2, 
	1, 2, 2, 4, 5, 2, 4, 6, 
	2, 4, 7, 2, 4, 8, 2, 4, 
	9, 2, 4, 10, 2, 4, 11, 2, 
	4, 12, 2, 4, 13, 2, 4, 14, 
	2, 4, 15, 2, 4, 16, 2, 4, 
	17, 2, 4, 18, 2, 4, 19, 2, 
	4, 20, 2, 4, 21, 2, 4, 22, 
	2, 4, 23, 2, 4, 24, 2, 4, 
	25, 2, 4, 26, 2, 4, 27, 2, 
	4, 28
};

static const short _Scanner_key_offsets[] = {
	0, 0, 3, 3, 5, 8, 8, 10, 
	14, 20, 24, 26, 27, 31, 31, 32, 
	34, 82, 84, 85, 86, 88, 89, 90, 
	91, 92, 100, 108, 118, 128, 132, 140, 
	141, 149, 155, 162, 170, 172, 173, 174, 
	177, 178, 179, 188, 197, 206, 215, 224, 
	233, 244, 252, 263, 273, 282, 291, 300, 
	309, 318, 327, 336, 344, 353, 362, 371, 
	380, 389, 397, 406, 415, 424, 433, 442, 
	451, 459, 468, 477, 486, 494, 503, 512, 
	521, 530, 539, 548, 549, 550, 554, 563, 
	572, 581, 590, 599, 608, 617, 627, 636, 
	645, 654, 663, 672, 681, 690, 700, 709, 
	718, 727, 736, 745, 754, 763, 772, 781, 
	790, 799, 808, 809
};

static const char _Scanner_trans_keys[] = {
	10, 34, 92, 10, 37, 10, 39, 92, 
	48, 57, 43, 45, 48, 57, 48, 57, 
	65, 70, 97, 102, 43, 45, 48, 57, 
	48, 57, 45, 10, 34, 92, 96, 42, 
	42, 47, 10, 33, 34, 35, 36, 37, 
	38, 39, 40, 41, 42, 43, 44, 45, 
	46, 47, 48, 58, 59, 60, 61, 62, 
	63, 64, 70, 73, 78, 84, 91, 92, 
	93, 94, 96, 98, 101, 102, 105, 110, 
	114, 119, 123, 124, 125, 126, 49, 57, 
	65, 122, 33, 126, 61, 10, 10, 37, 
	38, 42, 62, 62, 46, 95, 48, 57, 
	65, 90, 97, 122, 46, 95, 48, 57, 
	65, 90, 97, 122, 46, 69, 95, 101, 
	48, 57, 65, 90, 97, 122, 43, 45, 
	46, 95, 48, 57, 65, 90, 97, 122, 
	76, 105, 48, 57, 46, 95, 48, 57, 
	65, 90, 97, 122, 42, 46, 69, 76, 
	101, 105, 120, 48, 57, 69, 76, 101, 
	105, 48, 57, 46, 69, 76, 101, 105, 
	48, 57, 80, 112, 48, 57, 65, 70, 
	97, 102, 48, 57, 58, 58, 45, 60, 
	61, 61, 61, 46, 65, 95, 48, 57, 
	66, 90, 97, 122, 46, 76, 95, 48, 
	57, 65, 90, 97, 122, 46, 83, 95, 
	48, 57, 65, 90, 97, 122, 46, 69, 
	95, 48, 57, 65, 90, 97, 122, 46, 
	95, 110, 48, 57, 65, 90, 97, 122, 
	46, 95, 102, 48, 57, 65, 90, 97, 
	122, 46, 65, 85, 95, 97, 48, 57, 
	66, 90, 98, 122, 46, 95, 48, 57, 
	65, 90, 97, 122, 46, 95, 99, 105, 
	114, 48, 57, 65, 90, 97, 122, 46, 
	95, 104, 111, 48, 57, 65, 90, 97, 
	122, 46, 95, 97, 48, 57, 65, 90, 
	98, 122, 46, 95, 114, 48, 57, 65, 
	90, 97, 122, 46, 95, 97, 48, 57, 
	65, 90, 98, 122, 46, 95, 99, 48, 
	57, 65, 90, 97, 122, 46, 95, 116, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	101, 48, 57, 65, 90, 97, 122, 46, 
	95, 114, 48, 57, 65, 90, 97, 122, 
	46, 95, 48, 57, 65, 90, 97, 122, 
	46, 95, 109, 48, 57, 65, 90, 97, 
	122, 46, 95, 112, 48, 57, 65, 90, 
	97, 122, 46, 95, 108, 48, 57, 65, 
	90, 97, 122, 46, 95, 101, 48, 57, 
	65, 90, 97, 122, 46, 95, 120, 48, 
	57, 65, 90, 97, 122, 46, 95, 48, 
	57, 65, 90, 97, 122, 46, 95, 110, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	116, 48, 57, 65, 90, 97, 122, 46, 
	95, 101, 48, 57, 65, 90, 97, 122, 
	46, 95, 103, 48, 57, 65, 90, 97, 
	122, 46, 95, 101, 48, 57, 65, 90, 
	97, 122, 46, 95, 114, 48, 57, 65, 
	90, 97, 122, 46, 95, 48, 57, 65, 
	90, 97, 122, 46, 95, 101, 48, 57, 
	65, 90, 97, 122, 46, 95, 97, 48, 
	57, 65, 90, 98, 122, 46, 95, 108, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	48, 57, 65, 90, 97, 122, 46, 76, 
	95, 48, 57, 65, 90, 97, 122, 46, 
	76, 95, 48, 57, 65, 90, 97, 122, 
	46, 78, 95, 48, 57, 65, 90, 97, 
	122, 46, 82, 95, 48, 57, 65, 90, 
	97, 122, 46, 85, 95, 48, 57, 65, 
	90, 97, 122, 46, 69, 95, 48, 57, 
	65, 90, 97, 122, 91, 93, 10, 34, 
	92, 96, 46, 95, 114, 48, 57, 65, 
	90, 97, 122, 46, 95, 101, 48, 57, 
	65, 90, 97, 122, 46, 95, 97, 48, 
	57, 65, 90, 98, 122, 46, 95, 107, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	108, 48, 57, 65, 90, 97, 122, 46, 
	95, 115, 48, 57, 65, 90, 97, 122, 
	46, 95, 101, 48, 57, 65, 90, 97, 
	122, 46, 95, 111, 117, 48, 57, 65, 
	90, 97, 122, 46, 95, 114, 48, 57, 
	65, 90, 97, 122, 46, 95, 110, 48, 
	57, 65, 90, 97, 122, 46, 95, 99, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	116, 48, 57, 65, 90, 97, 122, 46, 
	95, 105, 48, 57, 65, 90, 97, 122, 
	46, 95, 111, 48, 57, 65, 90, 97, 
	122, 46, 95, 110, 48, 57, 65, 90, 
	97, 122, 46, 95, 102, 110, 48, 57, 
	65, 90, 97, 122, 46, 95, 101, 48, 
	57, 65, 90, 97, 122, 46, 95, 120, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	116, 48, 57, 65, 90, 97, 122, 46, 
	95, 101, 48, 57, 65, 90, 97, 122, 
	46, 95, 112, 48, 57, 65, 90, 97, 
	122, 46, 95, 101, 48, 57, 65, 90, 
	97, 122, 46, 95, 97, 48, 57, 65, 
	90, 98, 122, 46, 95, 116, 48, 57, 
	65, 90, 97, 122, 46, 95, 104, 48, 
	57, 65, 90, 97, 122, 46, 95, 105, 
	48, 57, 65, 90, 97, 122, 46, 95, 
	108, 48, 57, 65, 90, 97, 122, 46, 
	95, 101, 48, 57, 65, 90, 97, 122, 
	124, 0
};

static const char _Scanner_single_lengths[] = {
	0, 3, 0, 2, 3, 0, 0, 2, 
	0, 2, 0, 1, 4, 0, 1, 2, 
	44, 0, 1, 1, 2, 1, 1, 1, 
	1, 2, 2, 4, 4, 2, 2, 1, 
	6, 4, 5, 2, 0, 1, 1, 3, 
	1, 1, 3, 3, 3, 3, 3, 3, 
	5, 2, 5, 4, 3, 3, 3, 3, 
	3, 3, 3, 2, 3, 3, 3, 3, 
	3, 2, 3, 3, 3, 3, 3, 3, 
	2, 3, 3, 3, 2, 3, 3, 3, 
	3, 3, 3, 1, 1, 4, 3, 3, 
	3, 3, 3, 3, 3, 4, 3, 3, 
	3, 3, 3, 3, 3, 4, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 1, 0
};

static const char _Scanner_range_lengths[] = {
	0, 0, 0, 0, 0, 0, 1, 1, 
	3, 1, 1, 0, 0, 0, 0, 0, 
	2, 1, 0, 0, 0, 0, 0, 0, 
	0, 3, 3, 3, 3, 1, 3, 0, 
	1, 1, 1, 3, 1, 0, 0, 0, 
	0, 0, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 0, 0, 0, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 3, 3, 3, 3, 3, 3, 
	3, 3, 0, 0
};

static const short _Scanner_index_offsets[] = {
	0, 0, 4, 5, 8, 12, 13, 15, 
	19, 23, 27, 29, 31, 36, 37, 39, 
	42, 89, 91, 93, 95, 98, 100, 102, 
	104, 106, 112, 118, 126, 134, 138, 144, 
	146, 154, 160, 167, 173, 175, 177, 179, 
	183, 185, 187, 194, 201, 208, 215, 222, 
	229, 238, 244, 253, 261, 268, 275, 282, 
	289, 296, 303, 310, 316, 323, 330, 337, 
	344, 351, 357, 364, 371, 378, 385, 392, 
	399, 405, 412, 419, 426, 432, 439, 446, 
	453, 460, 467, 474, 476, 478, 483, 490, 
	497, 504, 511, 518, 525, 532, 540, 547, 
	554, 561, 568, 575, 582, 589, 597, 604, 
	611, 618, 625, 632, 639, 646, 653, 660, 
	667, 674, 681, 683
};

static const unsigned char _Scanner_indicies[] = {
	1, 2, 3, 0, 0, 4, 6, 5, 
	1, 8, 9, 7, 7, 10, 4, 12, 
	12, 10, 11, 13, 13, 13, 11, 15, 
	15, 16, 14, 16, 14, 18, 17, 4, 
	4, 20, 21, 19, 19, 23, 22, 23, 
	24, 22, 26, 27, 0, 28, 29, 5, 
	30, 7, 31, 32, 33, 34, 35, 36, 
	37, 38, 39, 41, 42, 43, 44, 45, 
	46, 47, 49, 50, 51, 52, 53, 1, 
	54, 55, 19, 56, 57, 58, 59, 60, 
	61, 62, 63, 64, 65, 66, 40, 48, 
	25, 4, 25, 68, 67, 69, 28, 70, 
	6, 5, 72, 71, 74, 73, 76, 75, 
	78, 77, 48, 48, 80, 48, 48, 79, 
	48, 48, 48, 48, 48, 4, 48, 81, 
	48, 81, 80, 48, 48, 79, 12, 12, 
	48, 48, 82, 48, 48, 79, 84, 85, 
	10, 83, 48, 48, 82, 48, 48, 79, 
	87, 86, 88, 89, 84, 89, 85, 90, 
	40, 83, 89, 84, 89, 85, 88, 83, 
	88, 89, 84, 89, 85, 40, 83, 92, 
	92, 13, 13, 13, 91, 16, 91, 94, 
	93, 96, 95, 98, 99, 100, 97, 102, 
	101, 104, 103, 48, 105, 48, 48, 48, 
	48, 79, 48, 106, 48, 48, 48, 48, 
	79, 48, 107, 48, 48, 48, 48, 79, 
	48, 108, 48, 48, 48, 48, 79, 48, 
	48, 109, 48, 48, 48, 79, 48, 48, 
	110, 48, 48, 48, 79, 48, 111, 112, 
	48, 113, 48, 48, 48, 79, 48, 115, 
	48, 48, 48, 114, 48, 48, 116, 117, 
	118, 48, 48, 48, 79, 48, 48, 119, 
	120, 48, 48, 48, 79, 48, 48, 121, 
	48, 48, 48, 79, 48, 48, 122, 48, 
	48, 48, 79, 48, 48, 123, 48, 48, 
	48, 79, 48, 48, 124, 48, 48, 48, 
	79, 48, 48, 125, 48, 48, 48, 79, 
	48, 48, 126, 48, 48, 48, 79, 48, 
	48, 127, 48, 48, 48, 79, 48, 128, 
	48, 48, 48, 79, 48, 48, 129, 48, 
	48, 48, 79, 48, 48, 130, 48, 48, 
	48, 79, 48, 48, 131, 48, 48, 48, 
	79, 48, 48, 132, 48, 48, 48, 79, 
	48, 48, 133, 48, 48, 48, 79, 48, 
	134, 48, 48, 48, 79, 48, 48, 135, 
	48, 48, 48, 79, 48, 48, 136, 48, 
	48, 48, 79, 48, 48, 137, 48, 48, 
	48, 79, 48, 48, 138, 48, 48, 48, 
	79, 48, 48, 139, 48, 48, 48, 79, 
	48, 48, 140, 48, 48, 48, 79, 48, 
	141, 48, 48, 48, 79, 48, 48, 142, 
	48, 48, 48, 79, 48, 48, 143, 48, 
	48, 48, 79, 48, 48, 144, 48, 48, 
	48, 79, 48, 145, 48, 48, 48, 79, 
	48, 146, 48, 48, 48, 48, 79, 48, 
	147, 48, 48, 48, 48, 79, 48, 148, 
	48, 48, 48, 48, 79, 48, 149, 48, 
	48, 48, 48, 79, 48, 150, 48, 48, 
	48, 48, 79, 48, 151, 48, 48, 48, 
	48, 79, 153, 152, 155, 154, 156, 156, 
	20, 21, 19, 48, 48, 157, 48, 48, 
	48, 79, 48, 48, 158, 48, 48, 48, 
	79, 48, 48, 159, 48, 48, 48, 79, 
	48, 48, 160, 48, 48, 48, 79, 48, 
	48, 161, 48, 48, 48, 79, 48, 48, 
	162, 48, 48, 48, 79, 48, 48, 163, 
	48, 48, 48, 79, 48, 48, 164, 165, 
	48, 48, 48, 79, 48, 48, 166, 48, 
	48, 48, 79, 48, 48, 167, 48, 48, 
	48, 79, 48, 48, 168, 48, 48, 48, 
	79, 48, 48, 169, 48, 48, 48, 79, 
	48, 48, 170, 48, 48, 48, 79, 48, 
	48, 171, 48, 48, 48, 79, 48, 48, 
	172, 48, 48, 48, 79, 48, 48, 173, 
	174, 48, 48, 48, 79, 48, 48, 175, 
	48, 48, 48, 79, 48, 48, 176, 48, 
	48, 48, 79, 48, 48, 177, 48, 48, 
	48, 79, 48, 48, 178, 48, 48, 48, 
	79, 48, 48, 179, 48, 48, 48, 79, 
	48, 48, 180, 48, 48, 48, 79, 48, 
	48, 181, 48, 48, 48, 79, 48, 48, 
	182, 48, 48, 48, 79, 48, 48, 183, 
	48, 48, 48, 79, 48, 48, 184, 48, 
	48, 48, 79, 48, 48, 185, 48, 48, 
	48, 79, 48, 48, 186, 48, 48, 48, 
	79, 188, 187, 1, 0
};

static const char _Scanner_trans_targs[] = {
	1, 0, 16, 2, 16, 3, 20, 4, 
	16, 5, 29, 16, 6, 35, 16, 10, 
	36, 16, 16, 12, 13, 85, 14, 15, 
	115, 17, 17, 18, 19, 16, 21, 16, 
	16, 22, 16, 16, 23, 25, 31, 32, 
	34, 37, 16, 39, 40, 41, 16, 16, 
	26, 42, 46, 48, 80, 83, 84, 16, 
	86, 90, 93, 101, 102, 105, 110, 16, 
	114, 16, 16, 16, 16, 16, 16, 16, 
	16, 16, 16, 16, 24, 16, 16, 16, 
	27, 28, 30, 16, 16, 16, 16, 16, 
	33, 7, 8, 16, 9, 16, 38, 16, 
	16, 16, 16, 11, 16, 16, 16, 16, 
	16, 43, 44, 45, 26, 47, 26, 49, 
	77, 79, 16, 50, 51, 66, 73, 52, 
	60, 53, 54, 55, 56, 57, 58, 59, 
	26, 61, 62, 63, 64, 65, 26, 67, 
	68, 69, 70, 71, 72, 26, 74, 75, 
	76, 26, 78, 26, 26, 81, 82, 26, 
	16, 16, 16, 16, 16, 87, 88, 89, 
	26, 91, 92, 26, 94, 95, 26, 96, 
	97, 98, 99, 100, 26, 26, 26, 103, 
	104, 26, 106, 107, 108, 109, 26, 111, 
	112, 113, 26, 16, 16
};

static const unsigned char _Scanner_trans_actions[] = {
	0, 0, 11, 0, 117, 0, 185, 0, 
	9, 0, 0, 111, 0, 7, 113, 0, 
	0, 115, 59, 0, 0, 179, 0, 0, 
	1, 191, 188, 0, 0, 25, 0, 35, 
	37, 0, 17, 63, 0, 0, 0, 182, 
	182, 0, 65, 7, 0, 0, 61, 27, 
	176, 0, 0, 0, 0, 0, 0, 19, 
	0, 0, 0, 0, 0, 0, 0, 31, 
	0, 33, 23, 87, 49, 109, 107, 93, 
	51, 85, 21, 81, 0, 105, 57, 71, 
	0, 176, 0, 75, 15, 13, 83, 67, 
	182, 0, 0, 77, 0, 89, 0, 91, 
	29, 101, 55, 0, 43, 79, 47, 103, 
	45, 0, 0, 0, 128, 0, 131, 0, 
	0, 0, 69, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	143, 0, 0, 0, 0, 0, 146, 0, 
	0, 0, 0, 0, 0, 137, 0, 0, 
	0, 140, 0, 122, 134, 0, 0, 125, 
	97, 39, 99, 41, 73, 0, 0, 0, 
	173, 0, 0, 167, 0, 0, 158, 0, 
	0, 0, 0, 0, 149, 161, 164, 0, 
	0, 170, 0, 0, 0, 0, 155, 0, 
	0, 0, 152, 95, 53
};

static const unsigned char _Scanner_to_state_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 3, 0, 
	119, 0, 0, 0, 0, 0, 0, 0, 
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
	0, 0, 0, 0
};

static const unsigned char _Scanner_from_state_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	5, 0, 0, 0, 0, 0, 0, 0, 
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
	0, 0, 0, 0
};

static const short _Scanner_eof_trans[] = {
	0, 0, 0, 5, 0, 0, 5, 12, 
	12, 15, 15, 18, 5, 5, 0, 0, 
	0, 5, 68, 70, 71, 72, 74, 76, 
	78, 80, 5, 80, 80, 84, 80, 87, 
	84, 84, 84, 92, 92, 94, 96, 98, 
	102, 104, 80, 80, 80, 80, 80, 80, 
	80, 115, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 153, 155, 157, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 188, 0
};

static const int Scanner_start = 16;
static const int Scanner_first_final = 16;
static const int Scanner_error = 0;

static const int Scanner_en_c_comment = 14;
static const int Scanner_en_main = 16;


#line 137 "lexer.rl"


void Parser::token( int tok, Value v)
{
	Parser::Result result;
	const char *data = ts;
	int len = te - ts;

	/*std::cout << '<' << tok << "> ";
	std::cout.write( data, len );
	std::cout << '\n';*/

	Parse(pParser, tok, v, &result); 
	
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

Parser::Parser(State& state) : line(0), col(0), have(0), state(state)
{
	
#line 463 "../parser.cpp"
	{
	cs = Scanner_start;
	ts = 0;
	te = 0;
	act = 0;
	}

#line 166 "lexer.rl"
}

int Parser::execute( const char* data, int len, bool isEof, Value& result)
{
	Result r;
	r.state = 0;

	pParser = ParseAlloc(malloc);

	const char *p = data;
	const char *pe = data+len;
	const char* eof = isEof ? pe : 0;

	
#line 486 "../parser.cpp"
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
	case 3:
#line 1 "NONE"
	{ts = p;}
	break;
#line 507 "../parser.cpp"
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
	case 0:
#line 29 "lexer.rl"
	{ {cs = 16; goto _again;} }
	break;
	case 4:
#line 1 "NONE"
	{te = p+1;}
	break;
	case 5:
#line 34 "lexer.rl"
	{act = 1;}
	break;
	case 6:
#line 36 "lexer.rl"
	{act = 3;}
	break;
	case 7:
#line 37 "lexer.rl"
	{act = 4;}
	break;
	case 8:
#line 38 "lexer.rl"
	{act = 5;}
	break;
	case 9:
#line 39 "lexer.rl"
	{act = 6;}
	break;
	case 10:
#line 40 "lexer.rl"
	{act = 7;}
	break;
	case 11:
#line 41 "lexer.rl"
	{act = 8;}
	break;
	case 12:
#line 42 "lexer.rl"
	{act = 9;}
	break;
	case 13:
#line 43 "lexer.rl"
	{act = 10;}
	break;
	case 14:
#line 44 "lexer.rl"
	{act = 11;}
	break;
	case 15:
#line 45 "lexer.rl"
	{act = 12;}
	break;
	case 16:
#line 46 "lexer.rl"
	{act = 13;}
	break;
	case 17:
#line 47 "lexer.rl"
	{act = 14;}
	break;
	case 18:
#line 48 "lexer.rl"
	{act = 15;}
	break;
	case 19:
#line 49 "lexer.rl"
	{act = 16;}
	break;
	case 20:
#line 50 "lexer.rl"
	{act = 17;}
	break;
	case 21:
#line 51 "lexer.rl"
	{act = 18;}
	break;
	case 22:
#line 52 "lexer.rl"
	{act = 19;}
	break;
	case 23:
#line 62 "lexer.rl"
	{act = 22;}
	break;
	case 24:
#line 64 "lexer.rl"
	{act = 23;}
	break;
	case 25:
#line 68 "lexer.rl"
	{act = 24;}
	break;
	case 26:
#line 124 "lexer.rl"
	{act = 65;}
	break;
	case 27:
#line 133 "lexer.rl"
	{act = 70;}
	break;
	case 28:
#line 134 "lexer.rl"
	{act = 71;}
	break;
	case 29:
#line 56 "lexer.rl"
	{te = p+1;{token( TOKEN_STR_CONST, Character::c(state, std::string(ts+1, te-ts-2)) );}}
	break;
	case 30:
#line 58 "lexer.rl"
	{te = p+1;{token( TOKEN_STR_CONST, Character::c(state, std::string(ts+1, te-ts-2)) );}}
	break;
	case 31:
#line 71 "lexer.rl"
	{te = p+1;{token( TOKEN_NUM_CONST, Complex::c(0, atof(std::string(ts, te-ts-1).c_str())) );}}
	break;
	case 32:
#line 74 "lexer.rl"
	{te = p+1;{token( TOKEN_NUM_CONST, Integer::c(atof(std::string(ts, te-ts-1).c_str())) );}}
	break;
	case 33:
#line 86 "lexer.rl"
	{te = p+1;{token( TOKEN_PLUS, Symbol(state, "+") );}}
	break;
	case 34:
#line 88 "lexer.rl"
	{te = p+1;{token( TOKEN_POW, Symbol(state, "^") );}}
	break;
	case 35:
#line 91 "lexer.rl"
	{te = p+1;{token( TOKEN_POW, Symbol(state, "^") );}}
	break;
	case 36:
#line 92 "lexer.rl"
	{te = p+1;{token( TOKEN_TILDE, Symbol(state, "~") );}}
	break;
	case 37:
#line 93 "lexer.rl"
	{te = p+1;{token( TOKEN_DOLLAR, Symbol(state, "$") );}}
	break;
	case 38:
#line 94 "lexer.rl"
	{te = p+1;{token( TOKEN_AT, Symbol(state, "@") );}}
	break;
	case 39:
#line 98 "lexer.rl"
	{te = p+1;{token( TOKEN_NS_GET_INT, Symbol(state, ":::") );}}
	break;
	case 40:
#line 101 "lexer.rl"
	{te = p+1;{token( TOKEN_LBRACE, Symbol(state, "{") );}}
	break;
	case 41:
#line 102 "lexer.rl"
	{te = p+1;{token( TOKEN_RBRACE, Symbol(state, "}") );}}
	break;
	case 42:
#line 103 "lexer.rl"
	{te = p+1;{token( TOKEN_LPAREN, Symbol(state, "(") );}}
	break;
	case 43:
#line 104 "lexer.rl"
	{te = p+1;{token( TOKEN_RPAREN, Symbol(state, ")") );}}
	break;
	case 44:
#line 106 "lexer.rl"
	{te = p+1;{token( TOKEN_LBB, Symbol(state, "[[") );}}
	break;
	case 45:
#line 108 "lexer.rl"
	{te = p+1;{token( TOKEN_RBB, Symbol(state, "]]") );}}
	break;
	case 46:
#line 111 "lexer.rl"
	{te = p+1;{token( TOKEN_LE, Symbol(state, "<=") );}}
	break;
	case 47:
#line 112 "lexer.rl"
	{te = p+1;{token( TOKEN_GE, Symbol(state, ">=") );}}
	break;
	case 48:
#line 113 "lexer.rl"
	{te = p+1;{token( TOKEN_EQ, Symbol(state, "==") );}}
	break;
	case 49:
#line 114 "lexer.rl"
	{te = p+1;{token( TOKEN_NE, Symbol(state, "!=") );}}
	break;
	case 50:
#line 115 "lexer.rl"
	{te = p+1;{token( TOKEN_AND2, Symbol(state, "&&") );}}
	break;
	case 51:
#line 116 "lexer.rl"
	{te = p+1;{token( TOKEN_OR2, Symbol(state, "||") );}}
	break;
	case 52:
#line 117 "lexer.rl"
	{te = p+1;{token( TOKEN_LEFT_ASSIGN, Symbol(state, "<-") );}}
	break;
	case 53:
#line 119 "lexer.rl"
	{te = p+1;{token( TOKEN_RIGHT_ASSIGN, Symbol(state, "->>") );}}
	break;
	case 54:
#line 120 "lexer.rl"
	{te = p+1;{token( TOKEN_LEFT_ASSIGN, Symbol(state, "<<-") );}}
	break;
	case 55:
#line 121 "lexer.rl"
	{te = p+1;{token( TOKEN_QUESTION, Symbol(state, "?") );}}
	break;
	case 56:
#line 127 "lexer.rl"
	{te = p+1;{token( TOKEN_COMMA );}}
	break;
	case 57:
#line 128 "lexer.rl"
	{te = p+1;{token( TOKEN_SEMICOLON );}}
	break;
	case 58:
#line 131 "lexer.rl"
	{te = p+1;{ {cs = 14; goto _again;} }}
	break;
	case 59:
#line 35 "lexer.rl"
	{te = p;p--;{token( TOKEN_NUM_CONST, Logical::NA );}}
	break;
	case 60:
#line 62 "lexer.rl"
	{te = p;p--;{token( TOKEN_SYMBOL, Symbol(state, std::string(ts, te-ts)) );}}
	break;
	case 61:
#line 64 "lexer.rl"
	{te = p;p--;{token( TOKEN_SYMBOL, Symbol(state, std::string(ts+1, te-ts-2)) );}}
	break;
	case 62:
#line 68 "lexer.rl"
	{te = p;p--;{token( TOKEN_NUM_CONST, Double::c(atof(std::string(ts, te-ts).c_str())) );}}
	break;
	case 63:
#line 82 "lexer.rl"
	{te = p;p--;{token( TOKEN_NUM_CONST );}}
	break;
	case 64:
#line 85 "lexer.rl"
	{te = p;p--;{token( TOKEN_EQ_ASSIGN, Symbol(state, "=") );}}
	break;
	case 65:
#line 87 "lexer.rl"
	{te = p;p--;{token( TOKEN_MINUS, Symbol(state, "-") );}}
	break;
	case 66:
#line 89 "lexer.rl"
	{te = p;p--;{token( TOKEN_DIVIDE, Symbol(state, "/") );}}
	break;
	case 67:
#line 90 "lexer.rl"
	{te = p;p--;{token( TOKEN_TIMES, Symbol(state, "*") );}}
	break;
	case 68:
#line 95 "lexer.rl"
	{te = p;p--;{token( TOKEN_NOT, Symbol(state, "!") );}}
	break;
	case 69:
#line 96 "lexer.rl"
	{te = p;p--;{token( TOKEN_COLON, Symbol(state, ":") );}}
	break;
	case 70:
#line 97 "lexer.rl"
	{te = p;p--;{token( TOKEN_NS_GET, Symbol(state, "::") );}}
	break;
	case 71:
#line 99 "lexer.rl"
	{te = p;p--;{token( TOKEN_AND, Symbol(state, "&") );}}
	break;
	case 72:
#line 100 "lexer.rl"
	{te = p;p--;{token( TOKEN_OR, Symbol(state, "|") );}}
	break;
	case 73:
#line 105 "lexer.rl"
	{te = p;p--;{token( TOKEN_LBRACKET, Symbol(state, "[") );}}
	break;
	case 74:
#line 107 "lexer.rl"
	{te = p;p--;{token( TOKEN_RBRACKET, Symbol(state, "]") );}}
	break;
	case 75:
#line 109 "lexer.rl"
	{te = p;p--;{token( TOKEN_LT, Symbol(state, "<") );}}
	break;
	case 76:
#line 110 "lexer.rl"
	{te = p;p--;{token( TOKEN_GT, Symbol(state, ">") );}}
	break;
	case 77:
#line 118 "lexer.rl"
	{te = p;p--;{token( TOKEN_RIGHT_ASSIGN, Symbol(state, "->") );}}
	break;
	case 78:
#line 124 "lexer.rl"
	{te = p;p--;{token(TOKEN_SPECIALOP, Symbol(state, std::string(ts, te-ts)) ); }}
	break;
	case 79:
#line 132 "lexer.rl"
	{te = p;p--;}
	break;
	case 80:
#line 68 "lexer.rl"
	{{p = ((te))-1;}{token( TOKEN_NUM_CONST, Double::c(atof(std::string(ts, te-ts).c_str())) );}}
	break;
	case 81:
#line 82 "lexer.rl"
	{{p = ((te))-1;}{token( TOKEN_NUM_CONST );}}
	break;
	case 82:
#line 109 "lexer.rl"
	{{p = ((te))-1;}{token( TOKEN_LT, Symbol(state, "<") );}}
	break;
	case 83:
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
	case 23:
	{{p = ((te))-1;}token( TOKEN_SYMBOL, Symbol(state, std::string(ts+1, te-ts-2)) );}
	break;
	case 24:
	{{p = ((te))-1;}token( TOKEN_NUM_CONST, Double::c(atof(std::string(ts, te-ts).c_str())) );}
	break;
	case 65:
	{{p = ((te))-1;}token(TOKEN_SPECIALOP, Symbol(state, std::string(ts, te-ts)) ); }
	break;
	case 70:
	{{p = ((te))-1;}token( TOKEN_NEWLINE );}
	break;
	default:
	{{p = ((te))-1;}}
	break;
	}
	}
	break;
#line 974 "../parser.cpp"
		}
	}

_again:
	_acts = _Scanner_actions + _Scanner_to_state_actions[cs];
	_nacts = (unsigned int) *_acts++;
	while ( _nacts-- > 0 ) {
		switch ( *_acts++ ) {
	case 1:
#line 1 "NONE"
	{ts = 0;}
	break;
	case 2:
#line 1 "NONE"
	{act = 0;}
	break;
#line 991 "../parser.cpp"
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

#line 180 "lexer.rl"

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


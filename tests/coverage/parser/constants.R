
NULL
NA
Inf 
NaN

TRUE
FALSE

1
10
0.1
.2
#NYI: printing with scientific notation
#1e-7
#1.2e+7

0xa
0xbcdef

0xA
0xBCDEF

1L
0x10L
1000000L
1e6L

#DIFF: R ignores the trailing L if the prefix is a float. Riposte truncates to an integer.
#1.1L
#1e-3L
#1.L

#NYI: complex
#2i
#4.1i
#1e-2i

'\''
"\""
'\n'
"\r"
'\t'
"\b"
'\a'
"\f"
'\v'
"\\"

#NYI: escape sequences
#'\000'
#"\123"
#'\456'
#"\x00"
#'\xab'

#'\u89ab'
#'\u{89ab}'

#"\U6789abcd"
#"\U{6789abcd}"

# A single quote may also be embedded directly in a double-quote delimited string and vice versa.
"R's parser"
'said "hello"'

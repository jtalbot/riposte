# Build grammar files. Not built with the rest of Riposte since this requires ragel on the path
# If you want to make fixes to the Riposte parser, install ragel.
default: parser

lexer: 
	ragel -o parser.cpp r.rl

lemon: lemon.c
	gcc -o $@ $<

parser: lexer lemon
	-./lemon -q r.y
	mv r.c r.cpp

clean:
	rm -rf lemon 

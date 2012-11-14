# Build grammar files. Requires ragel on the path

default: parser

lexer: 
	ragel -o lexer.cpp lexer.rl

lemon: lemon.c
	gcc -o $@ $<

parser: lexer lemon
	-./lemon -q grammar.y
	mv grammar.c grammar.cpp

clean:
	rm -rf lemon 

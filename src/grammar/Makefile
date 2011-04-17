# Build grammar files. Requires ragel on the path

default: parser

lexer: 
	ragel -o ../parser.cpp lexer.rl

lemon: lemon.c
	gcc -o $@ $<

parser: lexer lemon
	-./lemon -q parser.y
	mv parser.h ../tokens.h
	mv parser.c ../ast.cpp

clean:
	rm -rf lemon 

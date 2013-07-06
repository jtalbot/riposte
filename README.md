Riposte, a fast interpreter and JIT for R.

Justin Talbot <justintalbot@gmail.com>
Zach Devito 

We only do development on OSX and Linux. It's unlikely that our JIT will work on Windows.

Active development is currently happening on the [library branch](https://github.com/jtalbot/riposte/tree/library). See the Roadmap section below for information on development plans.


Compiling riposte
-----------------
1. Run `make release` in the riposte directory, or `make debug` for the debug version

2. Execute ./riposte to start


Flags
-----
-j # 		: start with # worker threads (defaults to 1, set to the number of cores on your machine)

-f filename	: execute R script

-v 		: verbose debug output for the vector trace recorder and JIT


Limitations
-----------
Riposte is in heavy development. Many things haven't been implemented and many things have been implemented incorrectly. Use with caution. Fixes welcome.


License
-------
Riposte is distributed under the BSD 2-clause license (http://www.opensource.org/licenses/bsd-license.php).

Riposte uses the following libraries:
* Eigen (http://eigen.tuxfamily.org) under the MPL2 license (http://www.mozilla.org/MPL/2.0/)
* Linenoise (https://github.com/antirez/linenoise/) under a BSD license


Testing
-------
Riposte comes with a growing set of validation tests that compare the output of Riposte to standard R. Run `make tests` to run all the tests (R must be installed). No output indicates all tests passed. Run `make tests/path/to/test` to execute a single test and print its diff.


Roadmap
-------
Riposte was developed as an academic research project with a focus on developing new techniques for executing dynamically typed vector code efficiently. We are now in the process of converting Riposte from research software to a robust drop-in replacement for R. This is a large effort; we estimate that the primary portion of this work will take about a year. Much of this work will be reimplementing R internal functions in a limited subset of R.

Riposte is currently considered in an alpha state. We'll move to a beta release when support for R's base library is complete, around the end of the year. Our current goal is to release version 1.0 next July.

Planned work for July-December 2013. The first three bullet points are currently in progress on the library branch. Partial work will be integrated to main by the end of July.

- [x] Load the standard base R library without errors
    - This will require support for about 15 primitive and external functions

- [ ] Support all R primitive operators (~200, 50 supported as of July 2013)
    - [x] The most common 40 or so will be appear as bytecodes in the Riposte VM, primarily control flow operators and a small set of common arithmetic
    - [ ] The rest will be implemented in the Riposte core library
    - [x] Implement new .Map, .Scan, or .Fold FFI functions to allow vector fusion through primitives implemented as external calls in the core library

- [ ] Support for the 200 most common internal functions (out of ~580, 30 supported as of July 2013)
    - They will be implemented primarily in R code in the Riposte internal library
    - To support internal functions which access interpreter state (such as the sys.frame, etc. functions), the Riposte VM will provide a small set of introspection bytecodes (roughly 10). The design goal is that all access to the interpreter state will go through the known bytecode API, allowing for better reasoning about side effects in external code.
    - All other C code necessary to support the internal functions will be implemented as part of the internal library, not as part of the interpreter itself.
    - [ ] Stretch goal: execute all example code from the R base library help files without error
    - Most of the internal functions will have to be implemented for this to happen

- [ ] Initial support for R's C-level API
    - This remains one of the big unknowns about Riposte. How much work will it be to support existing C libraries? 

- [ ] Convert Riposte's long vector JIT to use LLVM
    - This will come with a slight compilation performance penalty, but will be easier to use in the future and will be much more portable than our current solution.
    - Our short vector JIT already uses LLVM so this will make them easier to use together

- [ ] Integrate Riposte's short vector JIT from the tracing branch into the master branch

Next (January-June 2014)

- Complete support for all R internal functions

- Complete support for R's C-level API

- Further JIT performance improvements

- Code clean up

- Version 1.0 release July 2014
 

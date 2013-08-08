
getRegisteredNamespace <- NULL
setRegisteredNamespace <- NULL
getNamespaceRegistry <- NULL

(function() {
    namespaces <- .env_new(.getenv(.getenv(.getenv(NULL))))
    
    namespaces[['empty']] <- .getenv(.getenv(.getenv(NULL)))
    
    getRegisteredNamespace <<- function(name) {
        return( namespaces[[strip(name)]] )
    }

    setRegisteredNamespace <<- function(name, env) {
        namespaces[[strip(name)]] <- env

        if(!.env_exists(env, '.__NAMESPACE__.'))
            env[['.__NAMESPACE__.']] <- NULL
    
        env
    }

    getNamespaceRegistry <<- function() {
        namespaces
    }
})()

(function() {
    g <- globalenv()
    attr(g, 'name') <- 'R_GlobalEnv'
    
    e <- emptyenv()
    attr(e, 'name') <- 'R_EmptyEnv'
})()


.Platform <- list(
    OS.type="unix", 
    file.sep="/", 
    dynlib.ext=".so", 
    GUI="X11", 
    endian="little", 
    pkgType="mac.binary.leopard", 
    path.sep=":", 
    r_arch="x86_64"
    )

primitives <- .characters(
  
    # Arithmetic.R
    '+',
    '+.default',
    '-',
    '-.default',
    '*',
    '*.default',
    '/',
    '/.default',
    '^',
    '^.default',
    '%%',
    '%%.default',
    '%/%',
    '%/%.default',

    # Colon.R
    ':',

    # Comparison.R
    '<',
    '<.default',
    '>',
    '>.default',
    '<=',
    '<=.default',
    '>=',
    '>=.default',
    '==',
    '==.default',
    '==.environment',
    '!=',
    '!=.default',
 
    # Control.R
    'if',
    'for',
    'while',
    'repeat',
    'break',
    'next',

    # Encoding.R
    'enc2native',
    'enc2utf8',

    # Extract.R
    '[',
    '[.default',
    '[.environment',
    '[.matrix',
    '[.array',
    '[[',
    '[[.default',
    '[[.list',
    '[[.call',
    '[[.expression',
    '[[.pairlist',
    '[[.environment',
    '[[.closure',
    '[<-',
    '[<-.default',
    '[<-.environment',
    '[[<-',
    '[[<-.default',
    '[[<-.environment',
    '[[<-.closure',
    '$',
    '$<-',

    # Hyperbolic.R
    'cosh',
    'cosh.default',
    'sinh',
    'sinh.default',
    'tanh',
    'tanh.default',
    'acosh',
    'acosh.default',
    'asinh',
    'asinh.default',
    'atanh',
    'atanh.default',

    # Internal.R
    '.Internal',

    # Logic.R
    '!',
    '!.default',
    '&',
    '&.default',
    '|',
    '|.default',
    '&&',
    '||',

    # MathFun.R
    'abs',
    'abs.default',
    'sqrt',
    'sqrt.default',

    # NULL.R
    'is.null',

    # Paren.R
    '(',
    '{',

    # Round.R
    'floor',
    'floor.default',
    'ceiling',
    'ceiling.default',
    'trunc',
    'trunc.default',
    'round',
    'round.default',
    'signif',
    'signif.default',

    # Special.R
    'gamma',
    'gamma.default',
    'lgamma',
    'lgamma.default',
    'digamma',
    'digamma.default',
    'trigamma',
    'trigamma.default',

    # Trig.R
    'cos',
    'cos.default',
    'sin',
    'sin.default',
    'tan',
    'tan.default',
    'acos',
    'acos.default',
    'asin',
    'asin.default',
    'atan',
    'atan.default',
    'atan2',

    # all.R
    'all',
    'all.default',

    # any.R
    'any',
    'any.default',

    # array.R
    'is.array',
    'is.array.default',

    # as.environment.R
    'as.environment',

    # assignOps.R
    '=',
    '<-',
    '<<-',
    '->',
    '->>',

    # attr.R
    'attr',
    'attr<-',

    # attributes.R
    'attributes',
    'attributes<-',

    # base-internal.R
    '.subset',
    '.subset2',
    '.isMethodsDispatchOn',

    # browser.R
    'browser',

    # c.R
    'c',
    'c.default',

    # call.R
    'call',
    'is.call',
    'as.call',

    # character.R
    'as.character',
    'is.character',

    # class.R
    'class',
    'class<-',
    'unclass',
    'oldClass',
    'oldClass<-',

    # complex.R
    'Re',
    'Re.default',
    'Re.complex',
    'Im',
    'Im.default',
    'Im.complex',
    'Mod',
    'Mod.default',
    'Mod.complex',
    'Arg',
    'Arg.default',
    'Arg.complex',
    'Conj',
    'Conj.default',
    'Conj.complex',

    # cumsum.R
    'cumsum',
    'cumsum.default',
    'cumprod',
    'cumprod.default',
    'cummax',
    'cummax.default',
    'cummin',
    'cummin.default',

    # dim.R
    'dim',
    'dim.default',
    'dim<-',
    'dim<-.default',

    # dimnames.R
    'dimnames',
    'dimnames.default',
    'dimnames<-',
    'dimnames<-.default',

    # double.R
    'as.double',
    'is.double',

    # environment.R
    'environment<-',
    'is.environment',
    '.GlobalEnv',
    'globalenv',
    'emptyenv',
    'baseenv',

    # expression.R
    'expression',
    'is.expression',

    # function.R
    'function',
    'return',

    # integer.R
    'as.integer',
    'is.integer',

    # interactive.R
    'interactive',

    # invisible.R
    'invisible',

    # is.finite.R
    'is.finite',
    'is.finite.default',
    'is.infinite',
    'is.infinite.default',
    'is.nan',
    'is.nan.default',

    # is.function.R
    'is.function',

    # is.language.R
    'is.language',

    # is.na.R
    'is.na',

    # is.object.R
    'is.object',

    # is.raw.R
    'is.raw',
    'as.raw',

    # is.recursive.R
    'is.atomic',
    'is.recursive',

    # is.single.R
    'is.single',

    # length.R
    'length',
    'length.default',
    'length<-',
    'length<-.default',

    # levels.R
    'levels<-',
    'levels<-.default',

    # list.R
    'list',
    'is.list',
    'is.pairlist',

    # log.R
    'log',
    'log.default',
    'log10',
    'log10.default',
    'log2',
    'log2.default',
    'log1p',
    'log1p.default',
    'exp',
    'exp.default',
    'expm1',
    'expm1.default',

    # logical.R
    'as.logical',
    'is.logical',

    # matrix.R
    'is.matrix',

    # max.R
    'max',
    'max.default',

    # min.R
    'min',
    'min.default',

    # missing.R
    'missing',

    # mode.R
    'storage.mode<-',

    # name.R
    'is.symbol',

    # names.R
    'names',
    'names.default',
    'names<-',
    'names<-.default',

    # nargs.R
    'nargs',

    # nchar.R
    'nzchar',

    # numeric.R
    'as.numeric',
    'is.numeric',

    # on.exit.R
    'on.exit',

    # print.R (will be hidden by base)
    'print',
    'print.default',

    # proc.time
    'proc.time',

    # prod
    'prod',
    'prod.default',

    # range
    'range',
    'range.default',

    # rep
    'rep',
    'rep.default',

    # riposte.R
    'repl',
    'warnings', 
    'trace.config',
    'library',
    '::',
    'cummean',
    'cummean.default',
    'hypot',
    'time',
    'source',
    '__stop__',

    # s3.R
    'UseMethod',

    # seq.R
    'seq.int',
    'seq_along',
    'seq_len',

    # sign.R,
    'sign',
    'sign.default',

    # substitute.R
    'substitute',
    'quote',

    # sum.R
    'sum',
    'sum.default',

    # switch.R
    'switch',

    # tilde.R
    '~',

    # unlist.R
    'unlist.default',

    # zzz.R
    '.Platform'

    )

internals <- .characters(
    # Encoding.R
    'Encoding',
    'setEncoding',

    # R.Version.R
    'Version',

    # Recall.R
    'Recall',

    # Rhome.R
    'R.home',

    # Special.R
    'beta',
    'lbeta',
    'psigamma',
    'choose',
    'lchoose',

    # Sys.glob.R
    'Sys.glob',

    # args.R
    'args',

    # array.R
    'array',

    # as.function.R
    'as.function.default',

    # as.POSIX.R
    'as.POSIXct',
    'Date2POSIXlt',
    'as.POSIXlt',

    # assign.R
    'assign',

    # attach.R
    'attach',

    # basename.R
    'basename',
    'dirname',

    # browser.R
    'browserText',
    'browserCondition',
    'browserSetDebug',

    # cat.R
    'cat',

    # cbind.R
    'cbind',

    # charmatch.R
    'charmatch',

    # class.R
    'inherits',

    # col.R
    'col',

    # commandArgs.R
    'commandArgs',

    # comment.R
    'comment',
    'comment<-',

    # conditions.R
    '.addCondHands',
    '.signalCondition',

    # connections.R
    'file',
    'open',
    'close',
    'summary.connection',

    # data.frame.R
    'copyDFattr',

    # delayedAssign.R
    'delayedAssign',

    # deparse.R
    'deparse',

    # diag.R
    'diag',

    # do.call.R
    'do.call',

    # drop.R
    'drop',

    # duplicated.R
    'duplicated',
    'anyDuplicated',

    # environment.R
    'environment',
    'new.env',
    'parent.env',
    'parent.env<-',
    'environmentName',
    'env.profile',

    # eval.R
    'eval',

    # exists.R
    'exists',

    # file.R
    'file.exists',

    # file.info.R
    'file.info',

    # file.path.R
    'file.path',

    # formals.R
    'formals',

    # get.R
    'get',
    'mget',

    # gettext.R
    'gettext',
    'ngettext',
    'bindtextdomain',

    # getwd.R
    'getwd',
    'setwd',

    # grep.R
    'grepl',
    'grep',
    'regexpr',
    'gregexpr',
    'sub',
    'gsub',

    # identical.R
    'identical',

    # l10n_info.R
    'l10n_info',

    # lapply.R
    'lapply',
    'vapply',

    # list.R 
    'env2list',

    # locales.R
    'Sys.getlocale',
    'Sys.setlocale',

    # ls.R
    'ls',

    # mapply.R
    'mapply',

    # match.R
    'match',

    # match.call.R
    'match.call',

    # matrix.R
    'matrix',

    # mean.R
    'mean',

    # nchar.R
    'nchar',

    # normalizePath.R
    'normalizePath', 

    # ns-internals.R
    'isNamespaceEnv',

    # ns-load.R
    'getNamespaceRegistry',
    
    # ns-reflect.R (in zzz...)
    'getRegisteredNamespace',
    'setRegisteredNamespace',

    # options.R
    'options',

    # parse
    'parse',

    # paste.R
    'paste',
    'paste0',

    # path.expand.R
    'path.expand',

    # pmatch.R
    'pmatch',

    # print.R
    'print',
    'print.default',
    'print.function',

    # readLines.R
    'readLines',

    # rep.R
    'rep.int',
    'rep_len',

    # row.R
    'row',

    # s3.R
    'NextMethod',

    # search.R
    'search',

    # showConnections.R
    'stdin',
    'stdout',
    'stderr',

    # split.R
    'split',

    # sprintf.R
    'sprintf',

    # stop.R
    'stop',
    'geterrmessage',
    'seterrmessage',

    # strptime.R
    'strptime',

    # strsplit.R
    'strsplit',

    # substr.R
    'substr',
    'substr<-',

    # Sys.getenv
    'Sys.getenv',

    # Sys.setenv
    'Sys.setenv',
    'Sys.unsetenv',

    # sys.parent
    'sys.call',
    'sys.frame',
    'sys.nframe',
    'sys.function',
    'sys.nargs',
    'sys.parent',
    'sys.on.exit',
    'parent.frame',

    # typeof.R
    'typeof',

    # unique.R
    'unique',

    # unlist.R
    'unlist',
    'islistfactor',

    # vector.R
    'vector',
    'as.vector',
    'is.vector',

    # warning.R
    'warning',
    'printDeferredWarnings',

    # which.R
    'which',
    
    # which.min.R
    'which.min',
    'which.max'
    )

setRegisteredNamespace('core',
    .make.namespace('core', .getenv(NULL), primitives))
setRegisteredNamespace('internal',
    .make.namespace('internal', .getenv(NULL), internals))

.attach(.export('package:core', .getenv(NULL), primitives))

NULL

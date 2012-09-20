/*
* Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
This file implements the Rg runtime for managing GPU memory
(rgCopyToDevice*(), rgCopyFromDevice*(), rgAllocDvec()) and for
launching kernels on the GPU (rgLaunch()). 
*/

#include <sstream>
#include <string>
#include <cstdarg>
#include <vector>
#include <cstdio>
#include <iostream>
#include <fstream>

#include <cuda.h>
#include <builtin_types.h>

#include "rg_common.h"
#include "nvvm.h"

#include "drvapi_error_string.h"

#define checkCudaErrors(Err)  checkCudaErrors_internal (Err, __FILE__, __LINE__)

// controls verbosity of informational messages. Set through
int RgVerbosityLevel = 0;
bool RgDebugMode = false;

// error reporting function
void rgError(const char *Format,...) 
{
  char Buffer[1024];
  va_list ArgList;

  va_start(ArgList, Format);
  vsnprintf(Buffer, sizeof(Buffer), Format, ArgList);
  Rf_error(Buffer);
} // RgError

void rgVerbose(int Level, const char *Format,...)
{
  if (RgVerbosityLevel >= Level) {
    char Buffer[1024];
    va_list ArgList;

    va_start(ArgList, Format);
    vsnprintf(Buffer, sizeof(Buffer), Format, ArgList);
    REprintf("\nRG Info [%d] : %s", Level, Buffer);
  }
} // RgVerbose

// error check for CUDA calls
static void checkCudaErrors_internal(CUresult Err, const char *File, const int Line)
{
  if( CUDA_SUCCESS != Err) {
    std::stringstream Errstr;
    Errstr << "checkCudaErrors():: Driver API error = " << ((int)Err) << " \""
           << getCudaDrvErrorString(Err) << "\" from file <"
           << File << ">, line " << Line;
    std::string str = Errstr.str();
    rgError(str.c_str());
  }
} // __checkCudaErrors

// free device memory. Call back from Rg during Rg's garbage collection
extern "C" void freeDeviceMem(SEXP Expr)
{
  if (TYPEOF(Expr) != EXTPTRSXP)
    rgError("freeDeviceMem():: argument must an external pointer");
  CUdeviceptr *Mem = (CUdeviceptr *) R_ExternalPtrAddr(Expr);
  rgVerbose(5, "%s: freeing device memory addr: %p", __FUNCTION__, Mem);
  checkCudaErrors(cuMemFree(*Mem));
} // freeDeviceMem

// CUDA device initialization
static CUdevice cudaDeviceInit()
{
  rgVerbose(5, "%s: initializing CUDA", __FUNCTION__);

  CUdevice cuDevice = 0;
  int deviceCount = 0;
  CUresult err = cuInit(0);
  if (CUDA_SUCCESS == err)
    checkCudaErrors(cuDeviceGetCount(&deviceCount));
  if (deviceCount == 0) {
    std::stringstream errstr;
    errstr << "cudaDeviceInit():: no devices supporting CUDA";
    rgError(errstr.str().c_str());
  }
  checkCudaErrors(cuDeviceGet(&cuDevice, 0));
  char name[100];
  cuDeviceGetName(name, 100, cuDevice);

  int major=0, minor=0;
  checkCudaErrors( cuDeviceComputeCapability(&major, &minor, cuDevice) );
  if (major < 2) {
    rgError("cudaDeviceInit():: Device 0 is not sm_20 or later");
  }
  return cuDevice;
} // cudaDeviceInit

// CUDA initialization
static CUresult initCUDA()
{
  CUdevice dev = cudaDeviceInit();
  CUcontext context;
  checkCudaErrors(cuCtxCreate(&context, 0, dev));
  return CUDA_SUCCESS;
} // initCUDA

bool cudaInited = false;

// allocate a device vector of the give size and type.
static SEXP allocateOnDevice(unsigned elemSize, unsigned len, 
                             const char* typeName)
{
  unsigned memSize = elemSize*len;
  rgVerbose(5, "%s: memSize = %u, typeName = %s", __FUNCTION__,
            memSize, typeName);

  void *d_data = (void *)calloc(1, sizeof(CUdeviceptr));

  checkCudaErrors(cuMemAlloc((CUdeviceptr*)d_data, memSize));

  SEXP retval, aptr, alen, atype, rettmp;
  PROTECT(retval = Rf_allocList(3));
  rettmp = retval;
  PROTECT(aptr = R_MakeExternalPtr(d_data, R_NilValue, R_NilValue));
  R_RegisterCFinalizer(aptr, freeDeviceMem);

  SETCAR(rettmp, aptr);
  rettmp = CDR(rettmp);

  PROTECT(alen = Rf_allocVector(INTSXP, 1));
  INTEGER(alen)[0] = len;

  SETCAR(rettmp, alen);
  rettmp = CDR(rettmp);
  PROTECT(atype = Rf_mkString(typeName));
  SETCAR(rettmp, atype);

  UNPROTECT(4);
  return retval;
} // allocateOnDevice

// get raw pointer pointing to device memory
static SEXP getDvecDataPtr(SEXP v)
{
  if (!Rf_isList(v))
    rgError("getDvecDataPtr():: Invalid device vector passed");

  SEXP ptr = CAR(v);
  if (TYPEOF(ptr) != EXTPTRSXP) {
    rgError("getDvecDataPtr():: Invalid data pointer found on a device "
            "vector");
  }
  return ptr;
} // getDvecDataPtr


// get length of device vector
static SEXP getDvecLength(SEXP v)
{
  if (!Rf_isList(v))
    rgError("getDvecLength():: Invalid device vector passed");

  SEXP len = CAR(CDR(v));
  if (!Rf_isInteger(len))
    rgError("getDvecLength():: Invalid length found on a device vector");

  return len;
} // getDvecLength


// get type of device vector
static SEXP getDvecType(SEXP v)
{
  if (!Rf_isList(v))
    rgError("getDvecType():: Invalid device vector passed");

  SEXP type = CAR(CDR(CDR(v)));
  if (!Rf_isString(type))
    rgError("getDvecType():: Invalid type found on a device vector");

  return type;
} // getDvecType

static void *getStoragePointer(SEXP src)
{
  if (Rf_isInteger(src))
    return INTEGER(src);
  if (Rf_isReal(src))
    return REAL(src);
  if (Rf_isLogical(src))
    return LOGICAL(src);
  rgError("getStoragePointer():: Unknown vector type!");
  return NULL;
} // getStoragePointer

static void copyToDevice(SEXP dest, unsigned elemSize, unsigned n, SEXP src)
{
  unsigned memSize = elemSize * n;
  SEXP ptr = getDvecDataPtr(dest);
  CUdeviceptr *mem = (CUdeviceptr *) R_ExternalPtrAddr(ptr);
  checkCudaErrors(cuMemcpyHtoD(*mem, getStoragePointer(src), memSize));
  rgVerbose(5, "%s: memSize = %u", __FUNCTION__, memSize);
} // copyToDevice


// Note: invoked from rg.R
RGEXPORT SEXP rgCopyToDeviceDouble(SEXP a)
{
  if (!Rf_isReal(a))
    rgError("rgCopyToDeviceDouble():: Incompatible argument types!");
            
  unsigned len = LENGTH(a);
  SEXP ptr = allocateOnDevice(sizeof(double), len, "double");
  copyToDevice(ptr, sizeof(double), len, a);
  return ptr;
} // rgCopyToDeviceDouble


// Note: invoked from rg.R
RGEXPORT SEXP rgCopyToDeviceInteger(SEXP a)
{
  if (!Rf_isInteger(a))
    rgError("rgCopyToDeviceInteger():: Incompatible argument types!");
  
  unsigned len = LENGTH(a);
  SEXP ptr = allocateOnDevice(sizeof(int), len, "integer");
  copyToDevice(ptr, sizeof(int), len, a);
  return ptr;
} // rgCopyToDeviceInteger


// Note: invoked from rg.R
RGEXPORT SEXP rgCopyToDeviceLogical(SEXP a)
{
  if (!Rf_isLogical(a))
    rgError("rgCopyToDeviceLogical():: Incompatible argument types!");

  unsigned len = LENGTH(a);
  SEXP ptr = allocateOnDevice(sizeof(int), len, "logical");
  copyToDevice(ptr, sizeof(int), len, a);
  return ptr;
} // rgCopyToDeviceLogical


// Note: invoked from rg.R
RGEXPORT SEXP rgCopyFromDeviceDouble(SEXP a)
{
  if (!Rf_isList(a))
    rgError("rgCopyFromDeviceDouble():: Argument should be a list");

  SEXP ptr = getDvecDataPtr(a);
  SEXP len = getDvecLength(a);

  unsigned length = INTEGER(len)[0];
  CUdeviceptr *mem = (CUdeviceptr *) R_ExternalPtrAddr(ptr);
  unsigned memSize = length * sizeof(double);

  SEXP retval;
  PROTECT(retval = Rf_allocVector(REALSXP, length));
  double *h_data = REAL(retval);
  checkCudaErrors(cuMemcpyDtoH(h_data, *mem, memSize));
  UNPROTECT(1);

  rgVerbose(5, "%s: memSize = %u", __FUNCTION__, memSize);
  return retval;
} // rgCopyFromDeviceDouble


// Note: invoked from rg.R
RGEXPORT SEXP rgCopyFromDeviceInteger(SEXP a)
{
  if (!Rf_isList(a))
    rgError("rgCopyFromDeviceInteger():: Argument should be a list");

  SEXP ptr = getDvecDataPtr(a);
  SEXP len = getDvecLength(a);

  unsigned length = INTEGER(len)[0];
  CUdeviceptr *mem = (CUdeviceptr *) R_ExternalPtrAddr(ptr);
  unsigned memSize = length * sizeof(int);

  SEXP retval;
  PROTECT(retval = Rf_allocVector(INTSXP, length));
  int *h_data = INTEGER(retval);
  checkCudaErrors(cuMemcpyDtoH(h_data, *mem, memSize));
  UNPROTECT(1);
  rgVerbose(5, "%s: memSize = %u", __FUNCTION__, memSize);
  return retval;
} // rgCopyFromDeviceInteger


// Note: invoked from rg.R
RGEXPORT SEXP rgCopyFromDeviceLogical(SEXP a)
{
  if (!Rf_isList(a))
    rgError("rgCopyFromDeviceLogical():: Argument should be a list");

  SEXP ptr = getDvecDataPtr(a);
  SEXP len = getDvecLength(a);

  unsigned length = INTEGER(len)[0];
  CUdeviceptr *mem = (CUdeviceptr *) R_ExternalPtrAddr(ptr);
  unsigned memSize = length * sizeof(int);

  SEXP retval;
  PROTECT(retval = Rf_allocVector(LGLSXP, length));
  int *h_data = LOGICAL(retval);
  checkCudaErrors(cuMemcpyDtoH(h_data, *mem, memSize));
  UNPROTECT(1);
  rgVerbose(5, "%s: memSize = %u", __FUNCTION__, memSize);
  return retval;
} // rgCopyFromDeviceLogical


// allocate a device vector
// Note: invoked from rg.R
RGEXPORT SEXP rgAllocDvec(SEXP TypeNameExpr, SEXP LengthExpr)
{
  unsigned Length = *(INTEGER(LengthExpr));
  const char *TypeNameString = CHAR(STRING_ELT(TypeNameExpr, 0));
  rgVerbose(5, "%s: TypeName = %s, Length = %u", __FUNCTION__,
            TypeNameString, Length);

  if (strcmp(TypeNameString, "double") == 0) {
    return allocateOnDevice(sizeof(double), Length, TypeNameString);
  } else if (strcmp(TypeNameString, "integer") == 0) {
    return allocateOnDevice(sizeof(int), Length, TypeNameString);
  } else if (strcmp(TypeNameString, "logical") == 0) {
    return allocateOnDevice(sizeof(int), Length, TypeNameString);
  } else {
    rgError("rgAllocDvec():: unsupported type %s", TypeNameString);
    return R_NilValue;
  }
} // rgAllocDvec

// Note: invoked from rg.R
// set verbosity level for Rg informational messages. Return the
// old value for the verbosity level.
RGEXPORT SEXP rgSetVerbose(SEXP Expr)
{
  int Old = RgVerbosityLevel;
  RgVerbosityLevel = *(INTEGER(Expr));
  
  SEXP Retval;
  PROTECT(Retval = Rf_allocVector(INTSXP, 1));
  INTEGER(Retval)[0] = Old;
  UNPROTECT(1);
  return Retval;
} // rgSetVerbose

// Note: invoked from rg.R
// set debug mode on and off. 
// See rg.debug in rg.R for more details
RGEXPORT SEXP rgSetDebugMode(SEXP Expr)
{
  RgDebugMode = *(LOGICAL(Expr)) ? true : false;
  return Expr;
} // rgSetDebugMode

// Load the given PTX on the device. 
// Create a new device vector to store the result.
// Launch the kernel.
// Note: invoked from rg.R
RGEXPORT SEXP rgLaunch(SEXP PTXstr, SEXP Kernelname, SEXP ResultDV, 
                       SEXP ArgList)
{
  if (!Rf_isString(PTXstr))
    rgError("rgLaunch():: Expecting a string for the PTX argument");

  if (!Rf_isString(Kernelname))
    rgError("rgLaunch():: Expecting a string for the kernel name argument");

  const char *ptxstr = CHAR(STRING_ELT(PTXstr, 0));
  const char *kname = CHAR(STRING_ELT(Kernelname, 0));
  unsigned len = INTEGER(getDvecLength(ResultDV))[0];
  
  unsigned nthreads = 512;
  unsigned nblocks;
  if (nthreads >= len) {
    nthreads = len;
    nblocks = 1;
  }
  else {
    nblocks = 1 + (len-1)/nthreads;
  }
  CUmodule module;
  CUfunction kernel;

  rgVerbose(5, "%s: loading PTX on device", __FUNCTION__);
  checkCudaErrors(cuModuleLoadDataEx(&module, ptxstr, 0, 0, 0));
  rgVerbose(5, "%s: getting kernel handle, name: %s", __FUNCTION__, kname);
  checkCudaErrors(cuModuleGetFunction(&kernel, module, kname));
  rgVerbose(5, "%s: setting grid and thread configuration", __FUNCTION__);
  checkCudaErrors(cuFuncSetBlockShape(kernel, nthreads, 1, 1));

  std::vector<void *> args;
  args.push_back((void*)&len);
  args.push_back(R_ExternalPtrAddr(getDvecDataPtr(ResultDV)));

  for (SEXP CurNode = ArgList; CurNode != R_NilValue; CurNode = CDR(CurNode))
    args.push_back(R_ExternalPtrAddr(getDvecDataPtr(CAR(CurNode))));

  rgVerbose(5, "%s: launching kernel", __FUNCTION__);
  checkCudaErrors(cuLaunchKernel(kernel,
                                 nblocks, 1, 1, /* grid dim */
                                 nthreads, 1, 1, /* block dim */
                                 0, 0, /* shared mem, stream */
                                 &args[0],
                                 0));
  rgVerbose(5, "%s: device execution completed", __FUNCTION__);

  return ResultDV;
} // rgLaunch

// initialize Rg run time
// Note: invoked from rg.R
RGEXPORT SEXP rgInit()
{
  RgVerbosityLevel = 0;
  RgDebugMode = false;

  // initialize CUDA
  if (cudaInited == false) {
    checkCudaErrors(initCUDA());
    cudaInited = true;
  }
  
  // initialize NVVM
  nvvmResult ResCode;
  if ( (ResCode = nvvmInit()) == NVVM_SUCCESS) {
    int VersionMajor, VersionMinor;
    if ( (ResCode = nvvmVersion(&VersionMajor, &VersionMinor)) 
                                                      == NVVM_SUCCESS) 
    {
      REprintf("\nRg: Using NVVM version %d.%d\n", VersionMajor, VersionMinor);
    } else {
      rgError("Unable to get NVVM version information! Error Code: %d", 
              (int)ResCode);
    }
  } else {
    rgError("NVVM initialization failed! Error Code : %d", (int)ResCode);
  }

  SEXP Retval;
  PROTECT(Retval = Rf_allocVector(INTSXP, 1));
  INTEGER(Retval)[0] = 0;
  UNPROTECT(1);
  return Retval;
} // rgInit



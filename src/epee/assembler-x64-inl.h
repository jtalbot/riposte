// Copyright 2011 the V8 project authors. All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//     * Neither the name of Google Inc. nor the names of its
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef V8_X64_ASSEMBLER_X64_INL_H_
#define V8_X64_ASSEMBLER_X64_INL_H_

namespace v8 {
namespace internal {


// -----------------------------------------------------------------------------
// Utility functions

static inline bool is_intn(int x, int n)  {
  return -(1 << (n-1)) <= x && x < (1 << (n-1));
}

static inline bool is_int8(int x)  { return is_intn(x, 8); }
static inline bool is_int16(int x)  { return is_intn(x, 16); }
static inline bool is_int18(int x)  { return is_intn(x, 18); }
static inline bool is_int24(int x)  { return is_intn(x, 24); }

static inline bool is_uintn(int x, int n) {
  return (x & -(1 << n)) == 0;
}

static inline bool is_uint2(int x)  { return is_uintn(x, 2); }
static inline bool is_uint3(int x)  { return is_uintn(x, 3); }
static inline bool is_uint4(int x)  { return is_uintn(x, 4); }
static inline bool is_uint5(int x)  { return is_uintn(x, 5); }
static inline bool is_uint6(int x)  { return is_uintn(x, 6); }
static inline bool is_uint8(int x)  { return is_uintn(x, 8); }
static inline bool is_uint10(int x)  { return is_uintn(x, 10); }
static inline bool is_uint12(int x)  { return is_uintn(x, 12); }
static inline bool is_uint16(int x)  { return is_uintn(x, 16); }
static inline bool is_uint24(int x)  { return is_uintn(x, 24); }
static inline bool is_uint26(int x)  { return is_uintn(x, 26); }
static inline bool is_uint28(int x)  { return is_uintn(x, 28); }

static inline int NumberOfBitsSet(uint32_t x) {
  unsigned int num_bits_set;
  for (num_bits_set = 0; x; x >>= 1) {
    num_bits_set += x & 1;
  }
  return num_bits_set;
}

// -----------------------------------------------------------------------------
// Implementation of Assembler


void Assembler::emitl(uint32_t x) {
  *reinterpret_cast<uint32_t*>(pc_) = x;
  pc_ += sizeof(uint32_t);
}


void Assembler::emitq(uint64_t x) {
	*reinterpret_cast<uint64_t*>(pc_) = x;
  pc_ += sizeof(uint64_t);
}


void Assembler::emitw(uint16_t x) {
	*reinterpret_cast<uint16_t*>(pc_) = x;
  pc_ += sizeof(uint16_t);
}


void Assembler::emit_rex_64(Register reg, Register rm_reg) {
  emit(0x48 | reg.high_bit() << 2 | rm_reg.high_bit());
}


void Assembler::emit_rex_64(XMMRegister reg, Register rm_reg) {
  emit(0x48 | (reg.code() & 0x8) >> 1 | rm_reg.code() >> 3);
}


void Assembler::emit_rex_64(Register reg, XMMRegister rm_reg) {
  emit(0x48 | (reg.code() & 0x8) >> 1 | rm_reg.code() >> 3);
}


void Assembler::emit_rex_64(Register reg, const Operand& op) {
  emit(0x48 | reg.high_bit() << 2 | op.rex_);
}


void Assembler::emit_rex_64(XMMRegister reg, const Operand& op) {
  emit(0x48 | (reg.code() & 0x8) >> 1 | op.rex_);
}


void Assembler::emit_rex_64(Register rm_reg) {
  assert( (rm_reg.code() & 0xf) == rm_reg.code());
  emit(0x48 | rm_reg.high_bit());
}


void Assembler::emit_rex_64(const Operand& op) {
  emit(0x48 | op.rex_);
}


void Assembler::emit_rex_32(Register reg, Register rm_reg) {
  emit(0x40 | reg.high_bit() << 2 | rm_reg.high_bit());
}


void Assembler::emit_rex_32(Register reg, const Operand& op) {
  emit(0x40 | reg.high_bit() << 2  | op.rex_);
}


void Assembler::emit_rex_32(Register rm_reg) {
  emit(0x40 | rm_reg.high_bit());
}


void Assembler::emit_rex_32(const Operand& op) {
  emit(0x40 | op.rex_);
}


void Assembler::emit_optional_rex_32(Register reg, Register rm_reg) {
  byte rex_bits = reg.high_bit() << 2 | rm_reg.high_bit();
  if (rex_bits != 0) emit(0x40 | rex_bits);
}


void Assembler::emit_optional_rex_32(Register reg, const Operand& op) {
  byte rex_bits =  reg.high_bit() << 2 | op.rex_;
  if (rex_bits != 0) emit(0x40 | rex_bits);
}


void Assembler::emit_optional_rex_32(XMMRegister reg, const Operand& op) {
  byte rex_bits =  (reg.code() & 0x8) >> 1 | op.rex_;
  if (rex_bits != 0) emit(0x40 | rex_bits);
}


void Assembler::emit_optional_rex_32(XMMRegister reg, XMMRegister base) {
  byte rex_bits =  (reg.code() & 0x8) >> 1 | (base.code() & 0x8) >> 3;
  if (rex_bits != 0) emit(0x40 | rex_bits);
}


void Assembler::emit_optional_rex_32(XMMRegister reg, Register base) {
  byte rex_bits =  (reg.code() & 0x8) >> 1 | (base.code() & 0x8) >> 3;
  if (rex_bits != 0) emit(0x40 | rex_bits);
}


void Assembler::emit_optional_rex_32(Register reg, XMMRegister base) {
  byte rex_bits =  (reg.code() & 0x8) >> 1 | (base.code() & 0x8) >> 3;
  if (rex_bits != 0) emit(0x40 | rex_bits);
}


void Assembler::emit_optional_rex_32(Register rm_reg) {
  if (rm_reg.high_bit()) emit(0x41);
}


void Assembler::emit_optional_rex_32(const Operand& op) {
  if (op.rex_ != 0) emit(0x40 | op.rex_);
}


Address Assembler::target_address_at(Address pc) {
  return *reinterpret_cast<int32_t*>(pc) + pc + 4;
}


void Assembler::set_target_address_at(Address pc, Address target) {
  *reinterpret_cast<int32_t*>(pc) = static_cast<int32_t>(target - pc - 4);
  FlushICache(pc, sizeof(int32_t));
}

// -----------------------------------------------------------------------------
// Implementation of Operand

void Operand::set_modrm(int mod, Register rm_reg) {
  assert(is_uint2(mod));
  buf_[0] = mod << 6 | rm_reg.low_bits();
  // Set REX.B to the high bit of rm.code().
  rex_ |= rm_reg.high_bit();
}


void Operand::set_sib(ScaleFactor scale, Register index, Register base) {
  assert(len_ == 1);
  assert(is_uint2(scale));
  // Use SIB with no index register only for base rsp or r12. Otherwise we
  // would skip the SIB byte entirely.
  assert(!index.is(rsp) || base.is(rsp) || base.is(r12));
  buf_[1] = (scale << 6) | (index.low_bits() << 3) | base.low_bits();
  rex_ |= index.high_bit() << 1 | base.high_bit();
  len_ = 2;
}

void Operand::set_disp8(int disp) {
  assert(is_int8(disp));
  assert(len_ == 1 || len_ == 2);
  int8_t* p = reinterpret_cast<int8_t*>(&buf_[len_]);
  *p = disp;
  len_ += sizeof(int8_t);
}

void Operand::set_disp32(int disp) {
  assert(len_ == 1 || len_ == 2);
  int32_t* p = reinterpret_cast<int32_t*>(&buf_[len_]);
  *p = disp;
  len_ += sizeof(int32_t);
}


} }  // namespace v8::internal

#endif  // V8_X64_ASSEMBLER_X64_INL_H_

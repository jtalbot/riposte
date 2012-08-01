/*
 *  input.h
 *  
 *
 *  Created by dylanz on 8/1/12.
 *  Copyright 2012 Stanford University. All rights reserved.
 *
 */

#ifndef INPUT_H
#define INPUT_H

//===DEFINE INPUT VECTOR & KERNEL DIMENSIONS=====================================================================
//===============================================================================================================
#define VECTOR_LENGTH (1<<20)
#define TILE_WIDTH 128	
#define NUM_BLOCKS 180
#define NUM_VECT 300
#define MAX_SHARED_MEM (0xc000)

#endif
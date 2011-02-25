#include "rinst.h"
#include <time.h>
	
int r = 100000;
int vec_length = 1000;
void Re_ResetConsole()
{
}
void Re_FlushConsole()
{
}
void Re_ClearerrConsole()
{
}
void Re_ShowMessage(const char* mess){
  Re_WriteConsoleEx(mess,strlen(mess),0);
}

void Re_WriteConsoleEx(const char *buf1, int len, int oType){
  switch(oType){
  case 0:
    //fwrite(buf1,len,1,stdout);
    break;
  case 1:
    fwrite(buf1,len,1,stderr);
  }
}



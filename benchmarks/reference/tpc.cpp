#include<time.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "timing.h"

enum {
	ORDERKEY,
	PARTKEY,
	SUPPKEY,
	LINENUMBER,
	QUANTITY, //int
	EXTENDEDPRICE, //double
	DISCOUNT, //double
	TAX, //double
	RETURNFLAG, //char
	LINESTATUS, //??
	SHIPDATE, //date
	COMMITDATE,
	RECEIPTDATE,
	SHIPINSTRUCT,
	SHIPMODE,
	COMMENT,
	N_COLS
};


const char * return_item_codes = "ANR";
const char * line_status_codes = "FO";
const char * date_format = "%Y-%m-%d";
const char * time_for_filter = "1998-12-01";

time_t convert_time(const char * time) {
	struct tm t;
	memset(&t,0,sizeof(struct tm));
	const char * r = strptime(time,date_format,&t);
	assert(r != NULL);
	return mktime(&t);
}

static int interval = 90; //in days

//define N_ROWS for the correct problem size...
//static int N_ROWS = ;

const static int N_GROUPS = 6;

char intern_string(const char * fmt, const char * data) {
	for(int i = 0; *fmt; i++, fmt++) {
		if(*fmt == *data)
			return i;
	}
	assert(false);
	return 0;
}

double update_average(double cur, double new_value, double invnp1) {
	return cur + (new_value - cur) * invnp1;
}

int main() {
	int64_t * quantity = new int64_t[N_ROWS];
	double * extended_price = new double[N_ROWS];
	double * discount = new double[N_ROWS];
	double * tax = new double[N_ROWS];
	int64_t * group = new int64_t[N_ROWS];
	//char * return_flag = new char[N_ROWS];
	//char * line_status = new char[N_ROWS];
	int64_t * ship_date = new int64_t[N_ROWS];

	int ref_date = convert_time(time_for_filter);
	printf("%d\n", ref_date);	
	FILE * file = fopen("../data/lineitem.tbl","r");
	assert(file);
	for(int i = 0; i < N_ROWS; i++) {
		char buf[4096];
		char * columns[N_COLS];
		char * str = fgets(buf,4096,file);
		assert(str != NULL);
		columns[0] = strtok(buf,"|");
		for(int j = 1; j < N_COLS; j++) {
			columns[j] = strtok(NULL,"|");
			assert(columns[j]);
		}
		char * end = strtok(NULL,"\n");
		assert(end == NULL);
		quantity[i] = atoi(columns[QUANTITY]);
		extended_price[i] = atof(columns[EXTENDEDPRICE]);
		discount[i] = atof(columns[DISCOUNT]);
		tax[i] = atof(columns[TAX]);
		char return_flag = intern_string(return_item_codes,columns[RETURNFLAG]);
		char line_status = intern_string(line_status_codes, columns[LINESTATUS]);
		group[i] = (return_flag << 1) + line_status;
		ship_date[i] = convert_time(columns[SHIPDATE]);
	}
	
#define CREATE(typ, name) \
	typ name[N_GROUPS]; \
	memset(name,0,sizeof(typ) * N_GROUPS);
	
	CREATE(int64_t,sum_qty);
	CREATE(double,sum_base_price);
	CREATE(double,sum_disc_price);
	CREATE(double,sum_charge);
	CREATE(double,avg_qty);
	CREATE(double,avg_price);
	CREATE(double,avg_disc);
	CREATE(int64_t,count);

	double begin = current_time();

	for(int i = 0; i < N_ROWS; i++) {
		if(ship_date[i] <= ref_date - 24 * 60 * 60 * interval) {
			int grp = group[i];
			sum_qty[grp] += quantity[i];
			sum_base_price[grp] += extended_price[i];
			sum_disc_price[grp] += extended_price[i] * (1 - discount[i]);
			sum_charge[grp] += extended_price[i]*(1-discount[i])*(1+tax[i]);
			double invnp1 = 1.0/(count[grp]+1);
			avg_qty[grp] = update_average(avg_qty[grp],quantity[i],invnp1);
			avg_price[grp] = update_average(avg_price[grp],extended_price[i],invnp1);
			avg_disc[grp] = update_average(avg_disc[grp],discount[i],invnp1);
			count[grp]++;
		}
	}

	printf("Elapsed: %f\n", current_time()-begin);
	
#define REPORT(typ,name) do {\
	printf("%s = {",#name); \
	for(int i = 0; i < N_GROUPS; i++) { \
		PRINT_##typ(name[i]); \
	} \
	printf("}\n"); \
} while(0)

#define PRINT_int64_t(x) printf("%d,",(int)x)
#define PRINT_double(x) printf("%f, ",x)
	for(int i = 0; i < N_GROUPS; i++) {
		printf("%c %c\n", return_item_codes[i>>1], line_status_codes[i&1]);
	}
	REPORT(int64_t,sum_qty);
	REPORT(double,sum_base_price);
	REPORT(double,sum_disc_price);
	REPORT(double,sum_charge);
	REPORT(double,avg_qty);
	REPORT(double,avg_price);
	REPORT(double,avg_disc);
	REPORT(int64_t,count);
	
	
	return 0;
}

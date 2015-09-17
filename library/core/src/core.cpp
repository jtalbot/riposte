
#include <math.h>
#include <fstream>
#include <cstdio>

#include <pthread.h>

#include <eigen3/Eigen/Dense>

#include "../../../src/runtime.h"
#include "../../../src/compiler.h"
#include "../../../src/parser.h"
#include "../../../src/library.h"
#include "../../../src/coerce.h"


template<typename T>
T const& Cast(Value const& v) {
	if(v.type() != T::ValueType) _error("incorrect type passed to core function");
	return (T const&)v;
}

extern "C"
Value cat(State& state, Value const* args) {
	auto a = Cast<List>(args[0]);
	auto b = Cast<Character>(args[1]);
	for(int64_t i = 0; i < a.length(); i++) {
		if(!List::isNA(a[i])) {
            if(a[i].isEnvironment())
                printf("<environment: %llx>", (unsigned long long)a[i].p);
            else {
			    Character c = As<Character>(a[i]);
			    for(int64_t j = 0; j < c.length(); j++) {
                    if(c[j] == Character::NAelement)
                        printf("NA");
                    else
				        printf("%s", state.externStr(c[j]).c_str());
				    if(!(i == a.length()-1 && j == c.length()-1))
					    printf("%s", state.externStr(b[0]).c_str());
			    }
            }
		}
	}
	return Null();
}

extern "C"
Value library(State& state, Value const* args, Value& result) {
    auto dest = Cast<REnvironment>(args[0]);
	auto from = Cast<Character>(args[1]);
    if(from.length() > 0)
		loadPackage(state, dest.environment(), "library", state.externStr(from[0]));
    
    return Null();
}

extern "C"
Value readtable(State& state, Value const* args) {
	auto from = As<Character>(args[0]);
	auto sep_list = As<Character>(args[1]);
	auto format = As<Character>(args[2]);
	if(from.length() > 0 && sep_list.length() > 0 && format.length() > 0) {
		std::string name = state.externStr(from[0]);
		std::string sep = state.externStr(sep_list[0]);
		
		std::vector<void*> lists;
		
		for(int64_t i = 0; i < format.length(); i++) {
			if(Strings::Double == format[i] || Strings::Date == format[i]) {
				lists.push_back(new std::vector<double>);
			} else if(Strings::Character == format[i]) {
				lists.push_back(new std::vector<String>);
			}
			else if(Strings::NA !=format[i]) {
				_error("Unknown format specifier");
			}
		}
		FILE* file = fopen(name.c_str(), "r");
		if(file) {
			char buf[4096];
			for(int64_t line = 0;fgets(buf,4096,file);line++) {
				char * rest = buf;
				for(int64_t i = 0, list_idx = 0; i < format.length(); i++) {
					int sep_length = sep.length();
					char * sep_location = strstr(rest,sep.c_str());
					if(sep_location == NULL && i + 1 == format.length()) {
						sep_location = strstr(rest,"\n");
						sep_length = 1;
					}
					if(sep_location == NULL) {
						printf("line = %d, col = %d\n",(int)line, (int) (rest - buf));
						_error("Number of rows does not match format specifier");
					}
					*sep_location = '\0';
					if(Strings::Double == format[i]) {
						((std::vector<double>*)lists[list_idx])->push_back(atof(rest));
						list_idx++;
					} else if(Strings::Date == format[i]) {
						struct tm tm;
						memset(&tm,0,sizeof(struct tm));
						const char * result = strptime(rest,"%Y-%m-%d",&tm);
						if(result == NULL)
							_error("Value is not a date");
						double date = mktime(&tm);
						((std::vector<double>*)lists[list_idx])->push_back(date);
						list_idx++;
					} else if(Strings::Character == format[i]) {
						String s = MakeString(rest);
						((std::vector<String>*)lists[list_idx])->push_back(s);
						list_idx++;
					}
					rest = sep_location + sep_length;
				}
			}
			fclose(file);
		} else {
			_error("Unable to open file");
		}
		List l(lists.size());
		for(int64_t i = 0, list_idx = 0; i < format.length(); i++) {
			if(Strings::Double == format[i] || Strings::Date == format[i]) {
				std::vector<double> * data = (std::vector<double>*) lists[list_idx];
				Double r(data->size());
				for(uint64_t j = 0; j < data->size(); j++) {
					r[j] = (*data)[j];
				}
				l[list_idx] = r;
				delete data;
				list_idx++;
			} else if(Strings::Character == format[i]) {
				std::vector<String> * data = (std::vector<String>*) lists[list_idx];
				Character r(data->size());
				for(uint64_t j = 0; j < data->size(); j++) {
					r[j] = (*data)[j];
				}
				l[list_idx] = r;
				delete data;
				list_idx++;
			}
		}
		return l;
	} else {
		return Null();
	}
}


extern "C"
Value source(State& state, Value const* args) {
	auto file = Cast<Character>(args[0]);
	std::ifstream t(file[0]->s);
	std::stringstream buffer;
	buffer << t.rdbuf();
	std::string code = buffer.str();

	Value value;
	parse(state.global, file[0]->s, code.c_str(), code.length(), true, value);	
	
	return value;
}

extern "C"
Value paste(State& state, Value const* args) {
	auto a = As<Character>(args[0]);
	auto sep = As<Character>(args[1])[0];
	std::string r = "";
	for(int64_t i = 0; i < a.length()-1; i++) {
		r += a[i]->s;
		r += sep->s; 
	}
	if(0 < a.length()) r += a[a.length()-1]->s;
	return Character::c(MakeString(r));
}

#include <sys/time.h>

uint64_t readTime()
{
  timeval time_tt;
  gettimeofday(&time_tt, NULL);
  return (uint64_t)time_tt.tv_sec * 1000 * 1000 + (uint64_t)time_tt.tv_usec;
}

extern "C"
Value proctime(State& state, Value const* args) {
	uint64_t s = readTime();
	return Double::c(s/(1000000.0));
}

extern "C"
Value traceconfig(State & state, Value const* args) {
	auto c = As<Logical>(args[0]);
	if(c.length() == 0) _error("condition is of zero length");
	state.global.epeeEnabled = Logical::isTrue(c[0]);
	return Null();
}

// args( A, m, n, B, m, n )
extern "C"
Value matrixmultiply(State & state, Value const* args) {
	double mA = asReal1(args[1]);
	double nA = asReal1(args[2]);
	Eigen::MatrixXd aa = Eigen::Map<Eigen::MatrixXd>(As<Double>(args[0]).v(), mA, nA);
	
	double mB = asReal1(args[4]);
	double nB = asReal1(args[5]);
	Eigen::MatrixXd bb = Eigen::Map<Eigen::MatrixXd>(As<Double>(args[3]).v(), mB, nB);

	Double c(aa.rows()*bb.cols());
	Eigen::Map<Eigen::MatrixXd>(c.v(), aa.rows(), bb.cols()) = aa*bb;
	return c;
}

// args( A, m, n )
extern "C"
Value eigensymmetric(State & state, Value const* args) {
	double mA = asReal1(args[1]);
	double nA = asReal1(args[2]);
	Eigen::MatrixXd aa = Eigen::Map<Eigen::MatrixXd>(As<Double>(args[0]).v(), mA, nA);
	
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(aa);
	Double c(aa.rows()*aa.cols());
	Eigen::Map<Eigen::MatrixXd>(c.v(), aa.rows(), aa.cols()) = eigenSolver.eigenvectors();
	Double v(aa.rows());
	Eigen::Map<Eigen::MatrixXd>(v.v(), aa.rows(), 1) = eigenSolver.eigenvalues();
	
	List r(2);
	r[0] = v;
	r[1] = c;
	return r;
}

// args( A, m, n )
extern "C"
Value eigen(State & state, Value const* args) {
	/*double mA = asReal1(args[1]);
	double nA = asReal1(args[2]);
	Eigen::MatrixXd aa = Eigen::Map<Eigen::MatrixXd>(As<Double>(args[0]).v(), mA, nA);
	
	Eigen::EigenSolver<Eigen::MatrixXd> eigenSolver(aa);
	Double c(aa.rows()*aa.cols());
	Eigen::Map<Eigen::MatrixXd>(c.v(), aa.rows(), aa.cols()) = eigenSolver.eigenvectors();
	Double v(aa.rows());
	//Eigen::Map<Eigen::MatrixXd>(v.v(), aa.rows(), 1) = eigenSolver.eigenvalues();
	
	List r(2);
	r[0] = v;
	r[1] = c;
	result = r;*/
	throw("NYI: eigen");
}

extern "C"
Value commandArgs(State& state, Value const* args) {
	return state.global.arguments;
}


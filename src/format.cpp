
#include "common.h"

std::string rawToStr( unsigned char n )
{
	std::ostringstream result;
	result << std::hex << std::setfill('0') << std::setw(2) << (uint64_t)n;
	return result.str();
}

std::string intToStr( int64_t n )
{
    std::ostringstream result;
    result << n;
    return result.str();
}

std::string intToHexStr( uint64_t n )
{
    std::ostringstream result;
    result << "0x" << std::hex << n;
    return result.str();
}

std::string charToHexStr( char c )
{
    std::ostringstream result;
    result << "0x" << std::hex << (int)c;
    return result.str();
}

std::string doubleToStr( double n, uint64_t decimals, bool fixed )
{
    std::ostringstream result;
    if(std::isnan(n)) return std::string("NaN");
    else if(n == std::numeric_limits<double>::infinity()) return std::string("Inf");
    else if(n == -std::numeric_limits<double>::infinity()) return std::string("-Inf");
    if(n == 0) n = 0;   // handle the -0 case
    
    result << std::setprecision(decimals);
    if(fixed) result << std::fixed;
    result << n;
    return result.str();
}

std::string complexToStr( std::complex<double> n )
{
    std::ostringstream result;
    result << n.real();
    result.setf(std::ios::showpos);
    result << n.imag() << "i";
    return result.str();
}

int64_t strToInt( std::string const& s) {
    char* end;
	int64_t r = strtol( s.c_str(), &end, 0 );
    // TODO: should catch overflow from strtol
    if( *s.c_str() == '\0' || *end != '\0' )
        throw std::domain_error("strToInt");
	return r;
}

int64_t hexStrToInt( std::string const& s) {
	int64_t r;
	std::istringstream(s) >> std::hex >> r;
	return r;
}

int64_t octStrToInt( std::string const& s) {
	int64_t r;
	std::istringstream(s) >> std::oct >> r;
	return r;
}

double strToDouble( std::string const& s) {
    char* end;
	double r = strtod( s.c_str(), &end );
    if( *s.c_str() == '\0' || *end != '\0' )
        throw std::domain_error("strToDouble");
	return r;
}

